#!/bin/bash
#!/bin/bash

#!/bin/bash

set -e

# Load environment variables
source .env

# Function to check instance state
check_instance_state() {
    aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text
}

# Function to get instance IP
get_instance_ip() {
    aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

# Wait for instance to reach 'stopped' state if it's stopping
while true; do
    state=$(check_instance_state)
    if [ "$state" = "stopped" ]; then
        break
    elif [ "$state" = "running" ]; then
        break
    else
        echo "Waiting for instance to reach a stable state. Current state: $state"
        sleep 10
    fi
done

# Start the instance if it's not already running
if [ "$state" != "running" ]; then
    echo "Starting instance..."
    aws ec2 start-instances --instance-ids $INSTANCE_ID
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
fi

# Get the IP address with retry
echo "Waiting for IP address assignment..."
for i in {1..30}; do
    AWS_IP=$(get_instance_ip)
    if [ -n "$AWS_IP" ] && [ "$AWS_IP" != "None" ]; then
        echo "Instance IP: $AWS_IP"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Failed to get instance IP after 30 attempts. Exiting."
        exit 1
    fi
    sleep 10
done

echo "export AWS_IP=$AWS_IP" > ~/aws_ip.sh

# Update SSH config
echo "Updating SSH config..."
sed -i.bak '/Host aws-transcribe/,/UserKnownHostsFile \/dev\/null/d' ~/.ssh/config
cat >> ~/.ssh/config <<EOL

Host aws-transcribe
    HostName ${AWS_IP}
    User ubuntu
    IdentityFile ${AWS_KEY_PATH}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOL

# Ensure the mount directory exists
mkdir -p "${MOUNT_DIR}"

# Wait for SSH to become available
echo "Waiting for SSH to become available..."
for i in {1..30}; do
    if ssh -o ConnectTimeout=5 aws-transcribe exit 2>/dev/null; then
        echo "SSH is now available."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "SSH did not become available after 30 attempts. Exiting."
        exit 1
    fi
    sleep 10
done

# Mount remote directory
echo "Mounting remote directory..."
sshfs aws-transcribe:${REMOTE_DIR} ${MOUNT_DIR}

if [ $? -eq 0 ]; then
    echo "Remote directory successfully mounted at ${MOUNT_DIR}"
else
    echo "Failed to mount remote directory. Please check your connection and try again."
    exit 1
fi

# Start Docker containers
echo "Starting Docker containers..."
ssh aws-transcribe "cd ${REMOTE_DIR} && docker compose up --build -d"

echo "Environment setup complete. Docker containers are starting."
echo "You can now edit files in the remote volume locally at ${MOUNT_DIR}"
echo "To test the server, use: curl http://aws-transcribe:8000"
echo "To SSH into the instance, use: ssh aws-transcribe"
echo "To use the IP address in your current shell, run: source ~/aws_ip.sh"

