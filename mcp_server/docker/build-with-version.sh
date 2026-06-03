#!/bin/bash
# Script to build Docker image with proper version tagging
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

# Build the image with explicit mnemic version
echo "Building Docker image..."
docker build \
  --build-arg MCP_SERVER_VERSION="${MCP_VERSION}" \
  --build-arg MNEMIC_CORE_VERSION="${MNEMIC_CORE_VERSION}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --build-arg VCS_REF="${MCP_VERSION}" \
  -f Dockerfile \
  -t "mnemic/mnemic-mcp:${MCP_VERSION}" \
  -t "mnemic/mnemic-mcp:${MCP_VERSION}-mnemic-${MNEMIC_CORE_VERSION}" \
  -t "mnemic/mnemic-mcp:latest" \
  ..

echo ""
echo "Build complete!"
echo "  MCP Server Version: ${MCP_VERSION}"
echo "  Mnemic Core Version: ${MNEMIC_CORE_VERSION}"
echo "  Build Date: ${BUILD_DATE}"
echo ""
echo "Image tags:"
echo "  - mnemic/mnemic-mcp:${MCP_VERSION}"
echo "  - mnemic/mnemic-mcp:${MCP_VERSION}-mnemic-${MNEMIC_CORE_VERSION}"
echo "  - mnemic/mnemic-mcp:latest"
echo ""
echo "To inspect image metadata:"
echo "  docker inspect mnemic/mnemic-mcp:${MCP_VERSION} | jq '.[0].Config.Labels'"
