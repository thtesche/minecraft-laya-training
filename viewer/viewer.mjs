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
//          [--first-person] [--no-commands]

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
  commands: true
}

// Where the camera sits relative to MineAI: above it, so LayaView's own mesh does not cover it.
const OFFSET = { x: 0, y: 3, z: 0 }
// Blocks per follow step (every FOLLOW_MS). The server refuses a single move that is too long
// ("moved too quickly"), so a far target is reached over several steps.
const MAX_STEP = 6
const FOLLOW_MS = 100
// A lost target (out of tracking range, another dimension) is fetched with /tp at most this often.
const TP_EVERY_MS = 20000
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
  for (const key of ['port', 'view-port', 'view-distance']) opts[key] = Number(opts[key])
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
  let commandsAllowed = opts.commands
  let askedSpectator = false
  let lastTp = 0
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
    hint(`run /gamemode spectator ${opts.username} in your client (it stays that way for this world)`)
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
    const goal = target.position.offset(OFFSET.x, OFFSET.y, OFFSET.z)
    const delta = goal.minus(pos)
    const dist = delta.norm()
    if (dist < 0.05) return
    const step = dist > MAX_STEP ? delta.scaled(MAX_STEP / dist) : delta
    pos.add(step)
    bot.entity.yaw = target.yaw
    bot.entity.pitch = 0
    // A local move emits nothing; the viewer re-centres on 'move'.
    bot.emit('move')
  }

  bot.once('spawn', () => {
    reconnectMs = RECONNECT_MIN_MS
    log(`joined ${opts.host}:${opts.port} as ${opts.username}, gamemode ${bot.game.gameMode}`)
    mineflayerViewer(bot, {
      port: opts['view-port'],
      viewDistance: opts['view-distance'],
      firstPerson: opts['first-person']
    })
    log(`viewer on http://localhost:${opts['view-port']}`)
    ensureSpectator()
    followTimer = setInterval(follow, FOLLOW_MS)
  })

  bot.on('game', ensureSpectator)
  bot.on('respawn', ensureSpectator)

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
    try { bot.viewer?.close() } catch {}
    log(`disconnected (${reason}); reconnecting in ${reconnectMs / 1000}s`)
    setTimeout(start, reconnectMs)
    reconnectMs = Math.min(reconnectMs * 2, RECONNECT_MAX_MS)
  })
}

start()
