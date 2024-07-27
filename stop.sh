#!/bin/bash

set -e

# Load environment variables
source .env

echo "Starting cleanup process..."

# Function to get instance IP
get_instance_ip() {
    aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

# Get the current IP address of the instance
echo "Fetching instance IP..."
AWS_IP=$(get_instance_ip)

if [ -z "$AWS_IP" ]; then
    echo "Failed to get instance IP. The instance might be stopped or terminated."
    echo "Skipping file sync and Docker container stop, proceeding with cleanup..."
else
    # Stop Docker containers
    echo "Stopping Docker containers..."
    ssh aws-transcribe "cd ${REMOTE_DIR} && docker compose down"

    # Sync changed files to local directory
    echo "Syncing files from remote to local..."
    rsync -avz --exclude '.ds_store' --exclude '.git' --exclude '.venv' aws-transcribe:${REMOTE_DIR}/ "${PROJECT_DIR}/"
fi

# Unmount the remote directory
echo "Unmounting remote directory..."
if mount | grep -q "${MOUNT_DIR}"; then
    umount "${MOUNT_DIR}"
    echo "Remote directory unmounted successfully."
else
    echo "Remote directory was not mounted."
fi

# Remove the mount directory if it's empty
if [ -d "${MOUNT_DIR}" ] && [ -z "$(ls -A ${MOUNT_DIR})" ]; then
    rmdir "${MOUNT_DIR}"
    echo "Removed empty mount directory."
fi

# Stop the EC2 instance
echo "Stopping EC2 instance..."
aws ec2 stop-instances --instance-ids ${INSTANCE_ID}

# Remove the SSH config entry
sed -i.bak '/Host aws-transcribe/,/StrictHostKeyChecking no/d' ~/.ssh/config

echo "Cleanup process completed. The EC2 instance is stopping in the background."

