#!/bin/bash
# Script to build and push standalone Docker image with both Neo4j and FalkorDB drivers
# This script queries PyPI for the latest mnemic version and includes it in the image tag

set -e

# Get MCP server version from pyproject.toml
MCP_VERSION=$(grep '^version = ' ../pyproject.toml | sed 's/version = "\(.*\)"/\1/')

# Get latest mnemic version from PyPI
echo "Querying PyPI for latest mnemic version..."
MNEMIC_CORE_VERSION=$(curl -s https://pypi.org/pypi/mnemic/json | python3 -c "import sys, json; print(json.load(sys.stdin)['info']['version'])")
echo "Latest mnemic version: ${MNEMIC_CORE_VERSION}"

# Get build metadata
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Build the standalone image with explicit mnemic version
echo "Building standalone Docker image..."
docker build \
  --build-arg MCP_SERVER_VERSION="${MCP_VERSION}" \
  --build-arg MNEMIC_CORE_VERSION="${MNEMIC_CORE_VERSION}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --build-arg VCS_REF="${VCS_REF}" \
  -f Dockerfile.standalone \
  -t "mnemic/knowledge-graph-mcp:standalone" \
  -t "mnemic/knowledge-graph-mcp:${MCP_VERSION}-standalone" \
  -t "mnemic/knowledge-graph-mcp:${MCP_VERSION}-mnemic-${MNEMIC_CORE_VERSION}-standalone" \
  ..

echo ""
echo "Build complete!"
echo "  MCP Server Version: ${MCP_VERSION}"
echo "  Mnemic Core Version: ${MNEMIC_CORE_VERSION}"
echo "  Build Date: ${BUILD_DATE}"
echo "  VCS Ref: ${VCS_REF}"
echo ""
echo "Image tags:"
echo "  - mnemic/knowledge-graph-mcp:standalone"
echo "  - mnemic/knowledge-graph-mcp:${MCP_VERSION}-standalone"
echo "  - mnemic/knowledge-graph-mcp:${MCP_VERSION}-mnemic-${MNEMIC_CORE_VERSION}-standalone"
echo ""
echo "To push to DockerHub:"
echo "  docker push mnemic/knowledge-graph-mcp:standalone"
echo "  docker push mnemic/knowledge-graph-mcp:${MCP_VERSION}-standalone"
echo "  docker push mnemic/knowledge-graph-mcp:${MCP_VERSION}-mnemic-${MNEMIC_CORE_VERSION}-standalone"
echo ""
echo "Or push all tags:"
echo "  docker push --all-tags mnemic/knowledge-graph-mcp"
