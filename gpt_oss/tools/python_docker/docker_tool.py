"""
Python Code Execution Tool for GPT-OSS

This module provides a tool that allows the model to execute Python code in various
execution environments. It's one of the most powerful tools as it enables the model
to perform computations, data analysis, and programmatic problem-solving.

Prerequisites:
--------------
For Docker backend (recommended for safety):
    $ docker image pull python:3.11

Execution Backends:
-------------------
1. docker (RECOMMENDED, DEFAULT):
   - Executes code in an isolated Docker container (python:3.11)
   - Each invocation creates a fresh container (stateless)
   - Safe for untrusted code
   - Requires Docker daemon running

2. dangerously_use_uv:
   - Executes code locally using 'uv run' in a temporary directory
   - No isolation - code runs with same privileges as the process
   - Faster but UNSAFE for untrusted code
   - Useful for development/testing

3. dangerously_use_local_jupyter:
   - Executes code in a persistent local Jupyter kernel
   - Stateful: variables persist across invocations
   - No isolation - UNSAFE for untrusted code
   - Best for interactive data analysis workflows

Safety Considerations:
----------------------
- The docker backend is the only safe option for untrusted model-generated code
- The "dangerously" prefixed backends run code with full access to the host system
- Always use docker in production or when the model is exposed to untrusted inputs
- The local Jupyter backend is useful for research workflows where statefulness helps

Integration with Model:
-----------------------
The model learns to:
- Send Python code as the message content
- Interpret execution results and errors
- Iteratively refine code based on outputs
- Use print() statements to observe results (important for docker/uv backends)

Configuration:
--------------
Set via environment variable:
    export PYTHON_EXECUTION_BACKEND=docker  # or dangerously_use_uv, dangerously_use_local_jupyter
    export PYTHON_LOCAL_JUPYTER_CONNECTION_FILE=/path/to/kernel.json  # for jupyter backend
"""

import asyncio
import contextlib
import io
import os
import queue
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

import docker
from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)

from ..tool import Tool

# Global Docker client singleton (initialized on first use)
_docker_client = None

# Valid execution backend options
VALID_EXECUTION_BACKENDS = {
    "docker",  # Safe: runs in isolated containers
    "dangerously_use_uv",  # Unsafe: runs locally with uv
    "dangerously_use_local_jupyter",  # Unsafe: runs in local Jupyter kernel
}

# Determine which backend to use (from environment or default to docker)
_default_backend = os.environ.get("PYTHON_EXECUTION_BACKEND", "docker")
if _default_backend not in VALID_EXECUTION_BACKENDS:
    _default_backend = "docker"

PYTHON_EXECUTION_BACKEND = _default_backend


def call_python_script(script: str) -> str:
    """
    Execute Python code in an isolated Docker container (SAFE).

    This function:
    1. Creates a fresh Python 3.11 Docker container
    2. Transfers the script into the container via tar archive
    3. Executes the script and captures stdout/stderr
    4. Destroys the container

    Safety:
    - Each execution is isolated in a fresh container
    - Container is destroyed after execution (no state persists)
    - Code cannot access the host filesystem or network (unless Docker configured otherwise)
    - Safe for untrusted model-generated code

    Args:
        script: Python code to execute

    Returns:
        Combined stdout/stderr output from the script.
        If no output is produced, returns a warning message.

    Note:
        The container is stateless, so each invocation is independent.
        The model must include all necessary code in each script.
    """
    global _docker_client
    if _docker_client is None:
        # Initialize Docker client on first use
        _docker_client = docker.from_env()
        # Ensure python:3.11 image is available
        try:
            _docker_client.images.get("python:3.11")
        except docker.errors.ImageNotFound:
            _docker_client.images.pull("python:3.11")

    # 1. Create a temporary tar archive containing the script
    script_name = "script.py"
    tarstream = io.BytesIO()
    with tarfile.open(fileobj=tarstream, mode="w") as tar:
        script_bytes = script.encode("utf-8")
        tarinfo = tarfile.TarInfo(name=script_name)
        tarinfo.size = len(script_bytes)
        tar.addfile(tarinfo, io.BytesIO(script_bytes))
    tarstream.seek(0)

    # 2. Start the container (using 'sleep infinity' to keep it alive)
    container = _docker_client.containers.create(
        "python:3.11", command="sleep infinity", detach=True
    )
    try:
        container.start()
        # 3. Put the script into the container
        container.put_archive(path="/tmp", data=tarstream.read())
        # 4. Execute the script
        exec_result = container.exec_run(f"python /tmp/{script_name}")
        output = exec_result.output.decode("utf-8")
        if not output.strip():
            # No output means the model likely forgot to use print()
            output = "[WARN] No output available. Use print() to output anything to stdout to receive the output"
    finally:
        # Always clean up the container
        container.remove(force=True)
    return output


def call_python_script_with_uv(script: str) -> str:
    """
    Execute Python code locally using uv (UNSAFE - for development only).

    This function:
    1. Creates a temporary directory
    2. Writes the script to a file
    3. Executes it with 'uv run --no-project python script.py'
    4. Returns stdout or stderr

    Safety:
    - NO ISOLATION: Code runs with same privileges as the process
    - Can access filesystem, network, environment variables
    - DO NOT use for untrusted code
    - Useful for development/testing where Docker overhead is undesirable

    Args:
        script: Python code to execute

    Returns:
        stdout if execution succeeded, stderr if it failed
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(script)
        # Execute with uv (fast Python package/environment manager)
        exec_result = subprocess.run(
            ["uv", "run", "--no-project", "python", script_path],
            capture_output=True)
        return (
            exec_result.stdout.decode("utf-8")
            if exec_result.returncode == 0
            else exec_result.stderr.decode("utf-8")
        )


class LocalJupyterSession:
    """
    Stateful helper that proxies execution through a local Jupyter kernel (UNSAFE).

    This class manages a persistent Jupyter kernel where code execution maintains state
    across multiple invocations. Variables, imports, and function definitions persist
    between calls, enabling interactive workflows.

    Safety:
    - NO ISOLATION: Code runs in a local Jupyter kernel with full system access
    - DO NOT use for untrusted code
    - Best for research/development workflows where statefulness is desired

    Usage:
        session = LocalJupyterSession()
        result1 = session.execute("x = 42")
        result2 = session.execute("print(x)")  # Can access x from previous call
        session.close()
    """

    def __init__(
        self,
        connection_file: str | None = None,
        *,
        timeout: float = 120.0,
    ) -> None:
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The dangerously_use_local_jupyter backend requires the jupyter_client package to be installed."
            ) from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client: BlockingKernelClient
        self._km: KernelManager | None = None

        if connection_file:
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(
                    f"Cannot find Jupyter connection file at '{connection_path}'."
                )
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            # Ensure the connection is ready before executing.
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            km = KernelManager()
            km.start_kernel()
            client = km.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
            self._km = km
            self._owns_kernel = True

    def execute(self, code: str, *, timeout: float | None = None) -> str:
        """Execute code in the kernel, returning combined stdout/stderr output."""

        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for Jupyter kernel output.") from exc

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain the shell channel to capture final execution status.
        while True:
            try:
                reply = client.get_shell_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError(
                    "Timed out waiting for Jupyter kernel execution reply."
                ) from exc

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            if stdout:
                stdout = f"{stdout.rstrip()}\n{stderr}"
            else:
                stdout = stderr

        if not stdout.strip():
            stdout = (
                "[WARN] No output available. Use print() to output anything to stdout to "
                "receive the output"
            )

        return stdout

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        self.close()

class PythonTool(Tool):
    """
    Tool for executing Python code with configurable backends.

    This tool enables the model to run Python code for computations, data analysis,
    and problem-solving. It supports three execution backends with different
    safety/performance tradeoffs.

    Backends:
    - docker: Safe, isolated, stateless (DEFAULT)
    - dangerously_use_uv: Fast, unsafe, stateless
    - dangerously_use_local_jupyter: Fast, unsafe, stateful

    The tool integrates with the OpenAI Harmony message format:
    - Receives messages with Python code in content[0].text
    - Returns execution results as tool messages
    - Handles errors gracefully and returns them to the model

    Attributes:
        _execution_backend: Which backend to use for code execution
        _local_jupyter_connection_file: Path to Jupyter kernel connection file (if using Jupyter)
        _local_jupyter_timeout: Timeout for Jupyter kernel operations
        _jupyter_session: Active Jupyter session (if using Jupyter backend)
        _execution_lock: Lock to serialize Jupyter executions (prevents race conditions)
    """
    def __init__(
        self,
        name: str = "python",
        *,
        execution_backend: str | None = None,
        local_jupyter_connection_file: str | None = None,
        local_jupyter_timeout: float = 60.0,
    ):
        """
        Initialize the PythonTool with specified execution backend.

        Args:
            name: Tool name (must be "python")
            execution_backend: Which backend to use (docker, dangerously_use_uv, or
                             dangerously_use_local_jupyter). Defaults to PYTHON_EXECUTION_BACKEND.
            local_jupyter_connection_file: Path to Jupyter kernel connection file
                                          (for dangerously_use_local_jupyter backend)
            local_jupyter_timeout: Timeout in seconds for Jupyter kernel operations

        Raises:
            ValueError: If execution_backend is not valid
            AssertionError: If name is not "python"
        """
        assert name == "python"

        # Determine which backend to use
        backend = execution_backend or PYTHON_EXECUTION_BACKEND
        if backend not in VALID_EXECUTION_BACKENDS:
            raise ValueError(
                "execution_backend must be one of: "
                + ", ".join(sorted(VALID_EXECUTION_BACKENDS))
            )

        self._execution_backend = backend
        self._local_jupyter_connection_file = (
            local_jupyter_connection_file
            or os.environ.get("PYTHON_LOCAL_JUPYTER_CONNECTION_FILE")
        )
        self._local_jupyter_timeout = local_jupyter_timeout

        self._jupyter_session: LocalJupyterSession | None = None
        self._execution_lock: asyncio.Lock | None = None

        # Initialize Jupyter session if using that backend
        if self._execution_backend == "dangerously_use_local_jupyter":
            self._execution_lock = asyncio.Lock()  # Serialize executions
            self._jupyter_session = LocalJupyterSession(
                connection_file=self._local_jupyter_connection_file,
                timeout=self._local_jupyter_timeout,
            )

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        if self._execution_backend == "dangerously_use_local_jupyter":
            return """
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. Internet access for this session is UNKNOWN. Depends on the cluster.
            """.strip()

        return """
Use this tool to execute STATELESS Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you. You have to use print statements to access the output.

IMPORTANT: Your python environment is not shared between calls. You will have to pass your entire code each time.
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(), description=self.instruction, tools=[]
        )

    def _make_response(
        self,
        output: str,
        channel: str | None = None,
    ) -> Message:
        content = TextContent(text=output)
        return self.make_response(content=content, channel=channel)

    def make_response(
        self,
        content: Content,
        *,
        metadata: dict[str, Any] | None = None,
        author: Author | None = None,
        channel: str | None = None,
    ) -> Message:
        tool_name = self.get_tool_name()
        author = Author(role=Role.TOOL, name=f"{tool_name}")

        message = Message(
            author=author,
            content=[content],
        ).with_recipient("assistant")

        if channel:
            message = message.with_channel(channel)

        return message

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """
        Execute Python code and return the results.

        This method:
        1. Extracts the Python code from message.content[0].text
        2. Routes to the appropriate execution backend
        3. Handles timeouts and errors
        4. Returns a Message with the execution output

        Flow for different backends:
        - docker: Calls call_python_script() which creates a fresh container
        - dangerously_use_uv: Calls call_python_script_with_uv() which uses uv locally
        - dangerously_use_local_jupyter: Uses persistent Jupyter session with lock for serialization

        Args:
            message: Incoming message with Python code in content[0].text

        Yields:
            Message with execution results (stdout/stderr or error message)

        Note:
            For Jupyter backend, we use an asyncio.Lock to ensure executions are serialized
            since Jupyter kernels don't handle concurrent executions well.
        """
        # Extract the Python code from the message
        script = message.content[0].text
        channel = message.channel

        # Route to the appropriate backend
        if self._execution_backend == "docker":
            # Docker backend: safe, isolated execution
            output = call_python_script(script)
        elif self._execution_backend == "dangerously_use_uv":
            # UV backend: fast but unsafe local execution
            output = call_python_script_with_uv(script)
        elif self._execution_backend == "dangerously_use_local_jupyter":
            # Jupyter backend: stateful but unsafe local execution
            assert self._jupyter_session is not None
            lock = self._execution_lock
            if lock is not None:
                # Serialize executions to avoid Jupyter kernel race conditions
                async with lock:
                    try:
                        output = self._jupyter_session.execute(script)
                    except TimeoutError as exc:
                        output = f"[ERROR] {exc}"
            else:
                try:
                    output = self._jupyter_session.execute(script)
                except TimeoutError as exc:
                    output = f"[ERROR] {exc}"
        else:
            raise ValueError(
                f"Invalid PYTHON_EXECUTION_BACKEND: {self._execution_backend}"
            )

        # Yield the execution result as a tool message
        yield self._make_response(output, channel=channel)

    def close(self) -> None:
        if self._jupyter_session is not None:
            self._jupyter_session.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        self.close()
