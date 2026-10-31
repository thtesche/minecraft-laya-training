# syntax=docker/dockerfile:1
# Vanilla Minecraft 1.21.4 dedicated server for reproducible MineAI runs.
# The jar is fetched in a throwaway stage so the runtime image carries no build tools.
FROM oven/bun:1.4.4 AS jar
ARG MINECRAFT_VERSION=1.21.4
COPY download-server.mjs /opt/download-server.mjs
RUN bun /opt/download-server.mjs ${MINECRAFT_VERSION} /opt/server.jar

FROM eclipse-temurin:21-jre
ARG MINECRAFT_VERSION=1.21.4
ENV MINECRAFT_VERSION=${MINECRAFT_VERSION}
WORKDIR /data/minecraft
COPY --from=jar /opt/server.jar /opt/server.jar
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 25565
ENTRYPOINT ["/entrypoint.sh"]
