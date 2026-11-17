#!/bin/bash
# Write the properties the lab needs, then run the server. SIGTERM reaches java directly,
# so the world saves on `docker compose down`.
set -e

SEED=${SEED:-97996358}
EULA=${EULA:-false}
PORT=${PORT:-25565}

cat > /data/minecraft/server.properties <<EOF
server-port=$PORT
online-mode=false
spawn-protection=0
max-players=4
level-name=lab
level-seed=$SEED
EOF

# The server refuses to start without eula.txt; `EULA=true` is the lab's acceptance.
if [[ "$EULA" == "true" ]]; then
  echo "eula=true" > /data/minecraft/eula.txt
fi

# A FIFO on fd 3 gives the server a console the lab can write to, so commands such as
# `op LayaView` work without attaching to the container.
[[ -p /tmp/minecraft-input ]] || mkfifo /tmp/minecraft-input
exec 3<>/tmp/minecraft-input

java "-Xmx${JAVA_MEMORY:-2G}" -jar /opt/server.jar nogui <&3 &
java_pid=$!
trap 'kill "$java_pid" 2>/dev/null' TERM INT
wait "$java_pid"
