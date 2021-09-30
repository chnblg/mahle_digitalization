from paramiko import SSHClient
from scp import SCPClient
from pathlib import Path

def ssh_scp_files(ssh_host: str, ssh_port: int, ssh_user: str, ssh_password: str, 
                    source_volume: str, destination_volume: str):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname=ssh_host,
                port=ssh_port,
                username=ssh_user,
                password=ssh_password,
                look_for_keys=False)

    with SCPClient(ssh.get_transport()) as scp:
        scp.put(source_volume, recursive=True, remote_path=destination_volume)

def remove_files(filepath: str, filename: str):
    for p in Path(filepath).glob(filename):
        p.unlink()
