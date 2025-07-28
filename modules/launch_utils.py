# this scripts installs necessary requirements and launches main program in webui.py
import logging
import re
import subprocess
import os
import shutil
import sys
import importlib.util
import importlib.metadata
import platform
import json
import shlex
from functools import lru_cache
from typing import NamedTuple
from pathlib import Path

from modules import cmd_args, errors
from modules.paths_internal import script_path, extensions_dir, extensions_builtin_dir
from modules.timer import startup_timer
from modules import logging_config
from modules_forge import forge_version
from modules_forge.config import always_disabled_extensions


args, _ = cmd_args.parser.parse_known_args()
logging_config.setup_logging(args.loglevel)

python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
dir_repos = "repositories"

# Whether to default to printing command output
default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')


def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if not (major == 3 and minor == 11):
        import modules.errors

        modules.errors.print_error_explanation(
            f"""
            This program is tested with 3.11.9 Python, but you have {major}.{minor}.{micro}.
            If you encounter any error regarding unsuccessful package/library installation,
            please downgrade (or upgrade) to the latest version of 3.11 Python,
            and delete the current Python "venv" folder in WebUI's directory.

            Use --skip-python-version-check to suppress this warning
            """
        )


@lru_cache()
def commit_hash():
    try:
        return subprocess.check_output([git, "-C", script_path, "rev-parse", "HEAD"], shell=False, encoding='utf8').strip()
    except Exception:
        return "<none>"


def git_tag():
    return forge_version.version


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def is_installed(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            return False

        return spec is not None

    return dist is not None


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_pip(command, desc=None, live=default_command_live):
    if args.skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def check_run_python(code: str) -> bool:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    return result.returncode == 0


def git_fix_workspace(*args, **kwargs):
    raise NotImplementedError()


def run_git(*args, **kwargs):
    raise NotImplementedError()


def git_clone(*args, **kwargs):
    raise NotImplementedError()


def git_pull_recursive(dir):
    for subdir, _, _ in os.walk(dir):
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{script_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

        stdout = run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env).strip()
        if stdout:
            print(stdout)
    except Exception as e:
        errors.report(str(e))


def list_extensions(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n''', exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get('disabled_extensions', []) + always_disabled_extensions)
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if disable_all_extensions != 'none' or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_dir):
        return []

    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions]


def list_extensions_builtin(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n''', exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get('disabled_extensions', []))
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if disable_all_extensions != 'none' or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_builtin_dir):
        return []

    return [x for x in os.listdir(extensions_builtin_dir) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if not os.path.isdir(extensions_dir):
        return

    with startup_timer.subcategory("run extensions installers"):
        for dirname_extension in list_extensions(settings_file):
            logging.debug(f"Installing {dirname_extension}")

            path = os.path.join(extensions_dir, dirname_extension)

            if os.path.isdir(path):
                run_extension_installer(path)
                startup_timer.record(dirname_extension)

    if not os.path.isdir(extensions_builtin_dir):
        return

    with startup_timer.subcategory("run extensions_builtin installers"):
        for dirname_extension in list_extensions_builtin(settings_file):
            logging.debug(f"Installing {dirname_extension}")

            path = os.path.join(extensions_builtin_dir, dirname_extension)

            if os.path.isdir(path):
                run_extension_installer(path)
                startup_timer.record(dirname_extension)

    return


re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all rerqirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    """

    import importlib.metadata
    import packaging.version

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            if line.strip() == "":
                continue

            m = re.match(re_requirement, line)
            if m is None:
                return False

            package = m.group(1).strip()
            version_required = (m.group(2) or "").strip()

            if version_required == "":
                continue

            try:
                version_installed = importlib.metadata.version(package)
            except Exception:
                return False

            if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
                return False

    return True


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu128")
    torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url {torch_index_url}")
    xformers_package = os.environ.get('XFORMERS_PACKAGE', f"xformers==0.0.31.post1 --extra-index-url {torch_index_url}")

    clip_package = os.environ.get('CLIP_PACKAGE', "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip")

    gradio_package = os.environ.get("GRADIO_PACKAGE", "gradio==4.43.0 gradio_imageslider==0.0.20 gradio_rangeslider==0.0.6")
    requirements_file = os.environ.get('REQS_FILE', "requirements.txt")

    try:
        # the existence of this file is a signal to webui.sh/bat that webui needs to be restarted when it stops execution
        os.remove(os.path.join(script_path, "tmp", "restart"))
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')
    except OSError:
        pass

    if not args.skip_python_version_check:
        check_python_version()

    startup_timer.record("checks")

    tag = git_tag()

    print(f"Python {sys.version}")
    print(f"Version: {tag}")

    if args.reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)
        startup_timer.record("install torch")

    if not args.skip_torch_cuda_test:
        if not check_run_python("import torch; assert torch.cuda.is_available()"):
            raise RuntimeError("PyTorch is not able to access CUDA")
        startup_timer.record("torch GPU test")

    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")
        startup_timer.record("install clip")

    if args.xformers and (not is_installed("xformers") or args.reinstall_xformers):
        run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")
        startup_timer.record("install xformers")

    if not is_installed("ngrok") and args.ngrok:
        run_pip("install ngrok", "ngrok")
        startup_timer.record("install ngrok")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    if not is_installed("gradio"):
        run_pip(f"install {gradio_package}", "gradio")

    if not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")
        startup_timer.record("install requirements")

    if not args.skip_install:
        run_extensions_installers(settings_file=args.ui_settings_file)

    if args.update_all_extensions:
        git_pull_recursive(extensions_dir)
        startup_timer.record("update extensions")

    if not requirements_met(requirements_file):
        run_pip(f'install -r "{requirements_file}"', "requirements")
        startup_timer.record("enforce requirements")

    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)


def configure_for_tests():
    if "--api" not in sys.argv:
        sys.argv.append("--api")
    if "--ckpt" not in sys.argv:
        sys.argv.append("--ckpt")
        sys.argv.append(os.path.join(script_path, "test/test_files/empty.pt"))
    if "--skip-torch-cuda-test" not in sys.argv:
        sys.argv.append("--skip-torch-cuda-test")
    if "--disable-nan-check" not in sys.argv:
        sys.argv.append("--disable-nan-check")

    os.environ['COMMANDLINE_ARGS'] = ""


def configure_forge_reference_checkout(a1111_home: Path):
    """Set model paths based on an existing A1111 checkout."""
    class ModelRef(NamedTuple):
        arg_name: str
        relative_path: str

    refs = [
        ModelRef(arg_name="--ckpt-dir", relative_path="models/Stable-diffusion"),
        ModelRef(arg_name="--vae-dir", relative_path="models/VAE"),
        ModelRef(arg_name="--embeddings-dir", relative_path="embeddings"),
        ModelRef(arg_name="--lora-dir", relative_path="models/lora"),
        # Ref A1111 need to have sd-webui-controlnet installed.
        ModelRef(arg_name="--controlnet-dir", relative_path="models/ControlNet"),
        ModelRef(arg_name="--controlnet-preprocessor-models-dir", relative_path="extensions/sd-webui-controlnet/annotator/downloads"),
    ]

    for ref in refs:
        target_path = a1111_home / ref.relative_path
        if not target_path.exists():
            print(f"Path {target_path} does not exist. Skip setting {ref.arg_name}")
            continue

        if ref.arg_name in sys.argv:
            # Do not override existing dir setting.
            continue

        sys.argv.append(ref.arg_name)
        sys.argv.append(str(target_path))


def start():
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {shlex.join(sys.argv[1:])}")
    import webui
    if '--nowebui' in sys.argv:
        webui.api_only()
    else:
        webui.webui()

    from modules_forge import main_thread

    main_thread.loop()
    return


def dump_sysinfo():
    from modules import sysinfo
    import datetime

    text = sysinfo.get()
    filename = f"sysinfo-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d-%H-%M')}.json"

    with open(filename, "w", encoding="utf8") as file:
        file.write(text)

    return filename
