#!/bin/bash

echo "=================================="
echo "MQTT Server Setup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}This script requires sudo privileges${NC}"
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo -e "${RED}Cannot detect OS${NC}"
    exit 1
fi

echo "Detected OS: $OS"

# Stop mosquitto if running
echo ""
echo "Stopping existing Mosquitto service..."
sudo systemctl stop mosquitto 2>/dev/null || true

# Install Mosquitto MQTT Broker
echo ""
echo "Installing Mosquitto MQTT Broker..."

case $OS in
    ubuntu|debian)
        sudo apt-get update
        sudo apt-get install -y mosquitto mosquitto-clients
        ;;
    centos|rhel|fedora)
        sudo yum install -y mosquitto mosquitto-clients
        ;;
    *)
        echo -e "${RED}Unsupported OS: $OS${NC}"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Mosquitto installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install Mosquitto${NC}"
    exit 1
fi

# Backup original config
if [ -f /etc/mosquitto/mosquitto.conf ]; then
    sudo cp /etc/mosquitto/mosquitto.conf /etc/mosquitto/mosquitto.conf.backup
    echo -e "${GREEN}✓ Backed up original configuration${NC}"
fi

# Create Mosquitto configuration directory if it doesn't exist
sudo mkdir -p /etc/mosquitto/conf.d

# Remove old custom config if exists
sudo rm -f /etc/mosquitto/conf.d/custom.conf

# Create custom Mosquitto configuration
echo ""
echo "Configuring Mosquitto..."

sudo tee /etc/mosquitto/conf.d/custom.conf > /dev/null << 'EOF'
# Allow anonymous connections
allow_anonymous true

# Listen on all interfaces
listener 1883

# Persistence
persistence true
persistence_location /var/lib/mosquitto/

# Logging
log_dest file /var/log/mosquitto/mosquitto.log
log_type error
log_type warning
log_type notice
log_type information

# Connection messages
connection_messages true
EOF

echo -e "${GREEN}✓ Mosquitto configured${NC}"

# Ensure proper permissions
echo ""
echo "Setting permissions..."
sudo chown -R mosquitto:mosquitto /var/lib/mosquitto/ 2>/dev/null || true
sudo chown -R mosquitto:mosquitto /var/log/mosquitto/ 2>/dev/null || true
sudo chmod -R 755 /var/lib/mosquitto/ 2>/dev/null || true
sudo chmod -R 755 /var/log/mosquitto/ 2>/dev/null || true

# Test configuration
echo ""
echo "Testing configuration..."
if sudo mosquitto -c /etc/mosquitto/mosquitto.conf -t > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Configuration is valid${NC}"
else
    echo -e "${RED}✗ Configuration test failed${NC}"
    echo "Checking detailed error:"
    sudo mosquitto -c /etc/mosquitto/mosquitto.conf -t
    exit 1
fi

# Enable and start Mosquitto service
echo ""
echo "Starting Mosquitto service..."

sudo systemctl enable mosquitto
sudo systemctl restart mosquitto

# Wait for service to start
sleep 3

# Check service status
if sudo systemctl is-active --quiet mosquitto; then
    echo -e "${GREEN}✓ Mosquitto is running${NC}"
else
    echo -e "${RED}✗ Mosquitto failed to start${NC}"
    echo ""
    echo "Service status:"
    sudo systemctl status mosquitto --no-pager
    echo ""
    echo "Recent logs:"
    sudo journalctl -xeu mosquitto.service --no-pager -n 20
    exit 1
fi

# Check if port is listening
echo ""
echo "Checking if MQTT port is listening..."
sleep 1
if sudo netstat -tuln 2>/dev/null | grep -q ':1883' || sudo ss -tuln 2>/dev/null | grep -q ':1883'; then
    echo -e "${GREEN}✓ MQTT server is listening on port 1883${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Port 1883 might not be listening yet${NC}"
fi

# Configure firewall (if UFW is installed)
if command -v ufw &> /dev/null; then
    echo ""
    echo "Configuring firewall..."
    sudo ufw allow 1883/tcp 2>/dev/null || true
    echo -e "${GREEN}✓ Firewall configured (port 1883 opened)${NC}"
fi

# Get VM IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}')

echo ""
echo "=================================="
echo -e "${GREEN}MQTT Server Setup Complete!${NC}"
echo "=================================="
echo ""
echo "MQTT Broker is running at:"
echo "  • Local access:  localhost:1883"
echo "  • Remote access: $IP_ADDRESS:1883"
echo ""
echo "Quick Test:"
echo "  Terminal 1: mosquitto_sub -h localhost -t 'test/#' -v"
echo "  Terminal 2: mosquitto_pub -h localhost -t 'test/hello' -m 'Hello MQTT'"
echo ""
echo "Service Commands:"
echo "  • Status:  sudo systemctl status mosquitto"
echo "  • Start:   sudo systemctl start mosquitto"
echo "  • Stop:    sudo systemctl stop mosquitto"
echo "  • Restart: sudo systemctl restart mosquitto"
echo "  • Logs:    sudo journalctl -u mosquitto -f"
echo ""
echo "=================================="