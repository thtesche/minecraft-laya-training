# Laya viewer

A live browser view of the MineAI bot. A second client, **LayaView**, joins the same world as a
spectator and [prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) renders what
it loads at <http://localhost:3006>. It is a pure add-on: no change to `mlaya` or the MCP server,
and LayaView never fights, builds or chats.

## Start

```bash
cd viewer
npm install
node viewer.mjs > viewer.log 2>&1 &
```

Options (defaults in brackets): `--host` [127.0.0.1], `--port` [25565, the LAN port of the
world], `--version` [1.21.4], `--username` [LayaView], `--target` [MineAI], `--view-port` [3006],
`--view-distance` [6], `--first-person`, `--no-commands`.

The viewer reconnects by itself (5 s, doubling to 60 s) when the world closes or LayaView is
kicked, so it can stay running across world restarts; pass `--port` when the LAN port changes.

## Rules

- **LayaView must be a spectator.** Minecraft skips the night only when every non-spectator
  player sleeps. A LayaView in survival would keep the bot's `sleep` from ending the night, and
  the bed, the lit gate and the Nether crossing all wait on that night. On its first join the
  viewer sends `/gamemode spectator LayaView` itself; if the world refuses it the command
  (no permission), run it once in your own client - the game mode is saved with LayaView's
  player data for that world. Until then LayaView stands still and does not follow.
- **Follow without a pathfinder.** A spectator passes through blocks, and mineflayer sends its
  local position to the server every tick, so the viewer moves LayaView towards MineAI (three
  blocks above it) in steps of at most six blocks. When MineAI is out of tracking range or in
  another dimension, the viewer sends `/tp LayaView MineAI` at most every 20 s - or, without
  permission, logs the command for you to run.
- **Chat stays quiet.** The viewer only ever sends commands, never plain text. A `/tp` still
  shows up as an admin message (`[LayaView: Teleported ...]`) to operators, and the MCP server
  records that as a `player_message` for MineAI; `/gamerule logAdminCommands false` in your
  client silences it. `--no-commands` turns all commands off.

## In the browser

Third person by default: drag to orbit around the camera point above MineAI, scroll to zoom.
`--first-person` renders from LayaView's own eyes instead.

`/spectate MineAI` in the world is no substitute: the server moves the camera, but mineflayer
does not follow the camera packet, so the browser view would not move with it.
