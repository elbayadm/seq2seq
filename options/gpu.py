import subprocess

def exec_cmd(command):
    # return stdout, stderr output of a command
    return subprocess.Popen(command, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).communicate()


def get_gpu_memory(gpuid):
    # Get the current gpu usage ('cos sometimes oar mess up)
    result, _ = exec_cmd('nvidia-smi -i %d --query-gpu=memory.free \
                         --format=csv,nounits,noheader' % int(gpuid))
    result = int(result.strip())
    return result


