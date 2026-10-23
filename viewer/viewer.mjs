// Laya viewer: a live browser view of the MineAI bot.
//
// A second, observing client (LayaView) joins the same world and prismarine-viewer renders
// what it loads. LayaView never fights, builds or chats; the only things it may send are the
// few commands below, and only when the world grants it the permission.
//
// It has to be a spectator. Minecraft skips the night only when every non-spectator player
// sleeps, so a LayaView in survival would keep the bot's `sleep` from ending the night - and
// the bed, the gate and the crossing all wait on that night. A spectator also draws no mobs
// and passes through blocks, which is what lets it follow without a pathfinder: mineflayer
// sends `bot.entity.position` to the server on every tick, so the follow loop moves that
// position towards MineAI in bounded steps and the server moves the spectator with it.
//
// Usage: node viewer.mjs [--host 127.0.0.1] [--port 25565] [--version 1.21.4]
//          [--username LayaView] [--target MineAI] [--view-port 3006] [--view-distance 6]
//          [--distance 2] [--height 1] [--first-person] [--no-commands]

import mineflayer from 'mineflayer'
import viewerPkg from 'prismarine-viewer'

const { mineflayer: mineflayerViewer } = viewerPkg

const DEFAULTS = {
  host: '127.0.0.1',
  port: 25565,
  version: '1.21.4',
  username: 'LayaView',
  target: 'MineAI',
  'view-port': 3006,
  'view-distance': 6,
  'first-person': false,
  distance: 2,
  height: 1,
  commands: true
}

// The chase camera stands `--distance` behind MineAI (horizontally) and `--height` above it,
// looking at it.
// prismarine-viewer moves its camera only in first-person mode - its third-person camera is set
// on the first position update and never again - so the viewer always runs first person and
// places LayaView itself where a chase camera would stand.
// MineAI's eye height, the point the chase camera looks at, and the height prismarine-viewer
// adds to LayaView's position for its own camera.
const TARGET_EYES = 1.6
const VIEWER_EYES = 1.6
// Directions round the bot a chase spot is tried in, and the sampling step of the line of sight.
const CHASE_DIRECTIONS = 16
const LINE_STEP = 0.25
// Blocks per follow step (every FOLLOW_MS). The server refuses a single move that is too long
// ("moved too quickly"), so a far target is reached over several steps.
const MAX_STEP = 6
const FOLLOW_MS = 100
// A lost target (out of tracking range, another dimension) is fetched with /tp at most this often.
const TP_EVERY_MS = 20000
// How often the log reports where the camera is and how far MineAI is from it.
const STATUS_MS = 30000
// The wait after the spawn before the game mode is judged.
const SETTLE_MS = 3000
const RECONNECT_MIN_MS = 5000
const RECONNECT_MAX_MS = 60000

function parseArgs (argv) {
  const opts = { ...DEFAULTS }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--first-person') opts['first-person'] = true
    else if (arg === '--no-commands') opts.commands = false
    else if (arg.startsWith('--') && i + 1 < argv.length) opts[arg.slice(2)] = argv[++i]
  }
  for (const key of ['port', 'view-port', 'view-distance', 'distance', 'height']) opts[key] = Number(opts[key])
  return opts
}

const opts = parseArgs(process.argv.slice(2))

function log (...parts) {
  console.log(new Date().toISOString(), ...parts)
}

let reconnectMs = RECONNECT_MIN_MS

function start () {
  const bot = mineflayer.createBot({
    host: opts.host,
    port: opts.port,
    version: opts.version,
    username: opts.username,
    auth: 'offline',
    physicsEnabled: true
  })

  let followTimer = null
  let statusTimer = null
  let commandsAllowed = opts.commands
  let askedSpectator = false
  let lastTp = 0
  let spawned = false
  let lastHint = 0

  // The only way out to the server's chat: commands, never plain text.
  function command (text) {
    if (!commandsAllowed) return false
    if (!text.startsWith('/')) throw new Error(`refusing to chat: ${text}`)
    log('command', text)
    bot.chat(text)
    return true
  }

  function hint (text) {
    const now = Date.now()
    if (now - lastHint < 60000) return
    lastHint = now
    log('host action needed:', text)
  }

  function spectating () {
    return bot.game?.gameMode === 'spectator'
  }

  function ensureSpectator () {
    if (spectating()) {
      // No gravity and no collision for the observer: the follow loop owns the position.
      bot.physicsEnabled = false
      return
    }
    bot.physicsEnabled = true
    if (!askedSpectator && command(`/gamemode spectator ${opts.username}`)) {
      askedSpectator = true
      return
    }
    // The command just went out; its answer comes as the next game-mode change.
    if (askedSpectator && commandsAllowed) return
    hint(`run /gamemode spectator ${opts.username} in your client (it stays that way for this world)`)
  }

  // Whether the camera may sit at `v`: inside a solid block the renderer culls the faces seen
  // from within, and the rock around the bot disappears from the picture.
  function open (v) {
    const block = bot.blockAt(v.floored())
    return !block || block.boundingBox === 'empty'
  }

  function clearLine (a, b) {
    const d = b.minus(a)
    const n = Math.max(1, Math.ceil(d.norm() / LINE_STEP))
    for (let i = 1; i < n; i++) {
      if (!open(a.plus(d.scaled(i / n)))) return false
    }
    return true
  }

  // A chase spot in open air with a clear line to MineAI's eyes: the configured distance and
  // height first, fanning out round the bot from the side the camera is already on, then
  // closer and lower. In a one-wide tunnel there is often none: then null, and the camera takes MineAI's
  // own eyes.
  function chaseSpot (pos, tp, yaw, eyes) {
    let hx = pos.x - tp.x
    let hz = pos.z - tp.z
    if (Math.hypot(hx, hz) < 0.5) {
      // Right on top of it: start behind its facing.
      hx = Math.sin(yaw)
      hz = Math.cos(yaw)
    }
    const side = Math.atan2(hx, hz)
    const tries = []
    for (const scale of [1, 0.75, 0.5]) {
      for (const lift of [1, 0.5, 0]) {
        for (let k = 0; k < CHASE_DIRECTIONS; k++) {
          // Fan out from the current side: 0, +1, -1, +2, -2 ... steps.
          const turn = Math.ceil(k / 2) * (k % 2 ? 1 : -1) * (2 * Math.PI / CHASE_DIRECTIONS)
          const a = side + turn
          const r = opts.distance * scale
          tries.push(tp.offset(Math.sin(a) * r, opts.height * lift, Math.cos(a) * r))
        }
      }
    }
    for (const spot of tries) {
      const cam = spot.offset(0, VIEWER_EYES, 0)
      if (open(cam) && clearLine(cam, eyes)) return spot
    }
    return null
  }

  function follow () {
    if (!bot.entity || !spectating()) return
    const target = bot.players[opts.target]?.entity
    if (!target) {
      const now = Date.now()
      if (now - lastTp > TP_EVERY_MS && command(`/tp ${opts.username} ${opts.target}`)) {
        lastTp = now
      } else if (!commandsAllowed) {
        hint(`${opts.target} is out of sight; run /tp ${opts.username} ${opts.target} in your client`)
      }
      return
    }
    const pos = bot.entity.position
    const tp = target.position
    const eyes = tp.offset(0, TARGET_EYES, 0)
    // The camera spot, or null for MineAI's own eyes (always open air: MineAI stands there).
    const spot = opts['first-person'] ? null : chaseSpot(pos, tp, target.yaw, eyes)
    const goal = spot ?? tp.clone()
    const delta = goal.minus(pos)
    const dist = delta.norm()
    if (dist > 0.05) pos.add(dist > MAX_STEP ? delta.scaled(MAX_STEP / dist) : delta)
    if (!spot) {
      bot.entity.yaw = target.yaw
      bot.entity.pitch = target.pitch
    } else {
      // Look at MineAI's eyes, the way mineflayer's lookAt computes it.
      const d = eyes.minus(pos.offset(0, VIEWER_EYES, 0))
      bot.entity.yaw = Math.atan2(-d.x, -d.z)
      bot.entity.pitch = Math.atan2(d.y, Math.hypot(d.x, d.z))
    }
    // A local move emits nothing; the viewer re-centres on 'move'.
    bot.emit('move')
  }

  function status () {
    if (!bot.entity) return
    const fmt = (v) => `${v.x.toFixed(1)},${v.y.toFixed(1)},${v.z.toFixed(1)}`
    const target = bot.players[opts.target]?.entity
    const seen = target
      ? `${opts.target} at ${fmt(target.position)}, ${bot.entity.position.distanceTo(target.position).toFixed(1)} m away`
      : `${opts.target} not in sight`
    log(`camera ${fmt(bot.entity.position)} (${bot.game?.gameMode}, ${bot.game?.dimension}); ${seen}`)
  }

  bot.once('spawn', () => {
    reconnectMs = RECONNECT_MIN_MS
    log(`joined ${opts.host}:${opts.port} as ${opts.username}, gamemode ${bot.game.gameMode}`)
    mineflayerViewer(bot, {
      port: opts['view-port'],
      viewDistance: opts['view-distance'],
      firstPerson: true
    })
    log(`viewer on http://localhost:${opts['view-port']}`)
    spawned = true
    // The game mode and MineAI's entity arrive a moment after the spawn; asking at once sent
    // a /gamemode and a /tp the world did not need, each an admin message MineAI records.
    lastTp = Date.now()
    setTimeout(ensureSpectator, SETTLE_MS)
    followTimer = setInterval(follow, FOLLOW_MS)
    statusTimer = setInterval(status, STATUS_MS)
  })

  bot.on('game', () => { if (spawned) ensureSpectator() })
  bot.on('respawn', () => { if (spawned) ensureSpectator() })

  bot.on('messagestr', (text) => {
    // A refused command means this world gives LayaView no permission: stop sending any.
    if (/Unknown or incomplete command|You do not have permission|permission level/i.test(text)) {
      if (commandsAllowed) log('commands refused, switching to host hints:', text)
      commandsAllowed = false
    }
  })

  bot.on('kicked', (reason) => log('kicked:', JSON.stringify(reason)))
  bot.on('error', (err) => log('error:', err.message))

  bot.on('end', (reason) => {
    clearInterval(followTimer)
    clearInterval(statusTimer)
    try { bot.viewer?.close() } catch {}
    log(`disconnected (${reason}); reconnecting in ${reconnectMs / 1000}s`)
    setTimeout(start, reconnectMs)
    reconnectMs = Math.min(reconnectMs * 2, RECONNECT_MAX_MS)
  })
}

start()
