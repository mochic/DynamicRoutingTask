import os
import time
import yaml
import np_zro


def collect_script_results(hostname, port, foraging_id, output_dir) -> str:
    """TODO add this, should take like 5 minutes...

    Params
    ------
    foraging_id: str
        foraging_id associated with script results to gather

    Returns
    -------
    str
        path to results collected

    Notes
    -----
    - Requires output_dir to be accessible by both devices, 
    ie: some sort of shared drive
    """


def run_script(hostname, port, script_path, params_path, max_script_duration=60.0 * 60.0):
    """
    Params
    ------
    max_script_duration: float
        max duration for script to run in seconds, will stop waiting for script after that duration

    """
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    agent_proxy = np_zro.Proxy(hostname, port=port)
    agent_proxy.start_script(
        os.path.abspath(script_path),
        params=params,
    )

    start_time = time.time()
    while True:
        agent_proxy.is_script_running()

        if time.time() > start_time + max_script_duration:
            break

        time.sleep(30.0)  # 30s check interval


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("hostname", type=str)
    parser.add_argument("port", type=str)
    parser.add_argument("script_path", type=str)
    parser.add_argument("params_path", type=str)

    args = parser.parse_args()

    run_script(
        args.hostname,
        args.port,
        args.script_path,
        args.params_path,
    )
