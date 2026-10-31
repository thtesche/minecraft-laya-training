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

exec java "-Xmx${JAVA_MEMORY:-2G}" -jar /opt/server.jar nogui
