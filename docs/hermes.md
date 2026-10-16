# Build your Hermes agent

Use your coding agent to develop this project.
Use Hermes Agent to run the comparison agent that your team builds.

The starter registers one native Hermes tool, `challenge_compare`.
Its handler raises `NotImplementedError`.
It has no retrieval process, comparison result, agent prompt, or stopping policy.

## Set up a separate profile

Install Hermes Agent with the [official instructions](https://hermes-agent.nousresearch.com/docs/getting-started/installation).
Record the installed version with `hermes --version`.
Run the following commands from the starter repository root.

Use a separate profile directory for the challenge:

```bash
export HERMES_HOME="$PWD/data/hermes"
mkdir -p "$HERMES_HOME/plugins"
cp integrations/hermes/config.example.yaml "$HERMES_HOME/config.yaml"
cp -R integrations/hermes/nemotron-challenge "$HERMES_HOME/plugins/"
```

Keep `data/hermes` out of Git.
It can contain conversations and credentials.
If that directory exists, inspect its contents before you replace its configuration.

The config uses the official [named custom provider interface](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/integrations/providers.md#named-custom-providers).
It selects `nvidia/Nemotron-3.5-Super-VL` at `https://api.tokenfactory.nebius.com/v1`.
It declares tool and vision support.
It uses your `SUPER_API_KEY` through `key_env`.

Set `SUPER_API_KEY` in the starter's root `.env` file.
Use your own Nebius Token Factory account and key.

## Check registration

```bash
hermes plugins doctor integrations/hermes/nemotron-challenge --ci
hermes plugins enable nemotron-challenge
hermes plugins list
```

[Plugin Doctor](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/plugins/index.md#validate-with-plugin-doctor) checks discovery, imports, and registration.
It does not call `handle_compare` or check comparison quality.

Start Hermes from the shell that set `HERMES_HOME`:

```bash
uv run python - <<'PY'
import os
from dotenv import load_dotenv

load_dotenv(".env", override=False)
os.execvp("hermes", ["hermes", "chat", "--toolsets", "nemotron_challenge"])
PY
```

Copy your changed plugin into the profile.
Restart Hermes.

## Complete the agent milestone

Work with your coding agent in `integrations/hermes/nemotron-challenge/__init__.py`.
Change the schema if your chosen tool boundary needs different inputs.
Implement `handle_compare` and define its result format.
Keep `plugin.yaml` and `register()` consistent when you add or rename tools.
Your team chooses its evidence checks, tool boundaries, stopping rules, and behavior when a source is missing.
Connect these tools to your own saved project artifacts and pipeline.

Implement the `agent` stage in `src/nemotron_challenge/tasks.py` to run your agent.
Plugin registration does not implement that stage.

Check real replies for the expected model ID and tool call.
Check the tool result and the final response against saved evidence.
Test a request with missing evidence.
If your agent uses image inputs, check a real image request.
The capability declarations in the config do not prove that the endpoint accepts either request.

Use the [native plugin guide](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/plugins/index.md) for the plugin API.

Registration and config checks used Hermes [commit `6255980b`](https://github.com/NousResearch/hermes-agent/commit/6255980b191e87815be728dcae9a25a3f39a9308).
These checks made no model calls.
Run Plugin Doctor after a Hermes update.
