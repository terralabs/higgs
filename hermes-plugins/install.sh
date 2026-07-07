#!/bin/bash
# higgsfield plugin installer — fetches from terralabs/higgs on GitHub.
# Usage: curl -sSL https://raw.githubusercontent.com/terralabs/higgs/main/hermes-plugins/install.sh | sh
set -e

REPO="terralabs/higgs"
BRANCH="main"
BASE="https://raw.githubusercontent.com/${REPO}/${BRANCH}/hermes-plugins"
DEST="${HERMES_HOME:-$HOME/.hermes}/plugins/image_gen/higgsfield"

echo "Installing higgsfield plugin to: $DEST"
mkdir -p "$DEST"

# Fetch the two files
curl -fsSL "$BASE/image_gen/higgsfield/plugin.yaml" -o "$DEST/plugin.yaml"
curl -fsSL "$BASE/image_gen/higgsfield/__init__.py" -o "$DEST/__init__.py"

echo "Files installed:"
ls -la "$DEST"
echo
echo "Verifying SHA-256..."
EXPECTED_YAML="2f8f318e9e10349874ba29bf72f4b81ec29a50365ff3fae63e0fdcdadf4ff6bc"
EXPECTED_INIT="df9ab541848ba19cfaeeea4d305aaf9b2516c76407cb458e0d0b890f7ca7f647"
ACTUAL_YAML=$(shasum -a 256 "$DEST/plugin.yaml" | awk '{print $1}')
ACTUAL_INIT=$(shasum -a 256 "$DEST/__init__.py" | awk '{print $1}')
if [ "$ACTUAL_YAML" = "$EXPECTED_YAML" ] && [ "$ACTUAL_INIT" = "$EXPECTED_INIT" ]; then
    echo "  ✓ SHA-256 match"
else
    echo "  ✗ SHA mismatch"
    echo "    expected: $EXPECTED_YAML / $EXPECTED_INIT"
    echo "    actual:   $ACTUAL_YAML / $ACTUAL_INIT"
    exit 1
fi
echo
echo "=== Next steps ==="
echo "1. Enable the plugin:"
echo "     hermes plugins enable higgsfield"
echo
echo "2. Restart Hermes to pick it up:"
echo "     hermes restart"
echo
echo "3. Pick Higgsfield as the active image_gen provider:"
echo "     hermes tools"
echo "     # -> Image Generation -> Higgsfield"
echo
echo "4. Test:"
echo "     hermes -z 'Generate an image of a neon coral geometry on a monolith, 16:9, no text'"
