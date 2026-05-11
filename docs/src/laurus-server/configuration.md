# Configuration

The laurus-server can be configured through CLI arguments, environment variables, and a TOML configuration file.

## Configuration Priority

Server and index settings are resolved in the following order (highest priority first):

```text
CLI arguments > Environment variables > Config file > Defaults
```

Log verbosity is controlled exclusively by the `RUST_LOG` environment variable (default: `info`).

For example:

```bash
# CLI argument wins over environment variable and config file
LAURUS_PORT=4567 laurus serve --config config.toml --port 1234
# -> Listens on port 1234

# Environment variable wins over config file
LAURUS_PORT=4567 laurus serve --config config.toml
# -> Listens on port 4567

# Config file value is used when no CLI argument or env var is set
laurus serve --config config.toml
# -> Uses port from config.toml (or default 50051 if not set)
```

## TOML Configuration File

### Format

```toml
[server]
host = "0.0.0.0"
port = 50051
http_port = 8080  # Optional: enables HTTP Gateway

[index]
data_dir = "./laurus_data"
```

Log verbosity is controlled by the `RUST_LOG` environment variable (default: `info`), not through the config file.

### Field Reference

#### `[server]` Section

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `host` | String | `"0.0.0.0"` | Listen address for the gRPC server |
| `port` | Integer | `50051` | Listen port for the gRPC server |
| `http_port` | Integer | -- | HTTP Gateway port. When set, the HTTP/JSON gateway starts alongside gRPC. |

#### `[index]` Section

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_dir` | String | `"./laurus_data"` | Path to the index data directory |

## Environment Variables

| Variable | Maps To | Description |
| :--- | :--- | :--- |
| `LAURUS_HOST` | `server.host` | Listen address |
| `LAURUS_PORT` | `server.port` | gRPC listen port |
| `LAURUS_HTTP_PORT` | `server.http_port` | HTTP Gateway port |
| `LAURUS_INDEX_DIR` | `index.data_dir` | Index data directory |
| `RUST_LOG` | -- | Log filter directive (e.g. `info`, `debug`, `laurus=debug,tonic=warn`) |
| `LAURUS_CONFIG` | -- | Path to TOML config file |

## CLI Arguments

| Option | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config <PATH>` | `-c` | -- | Path to TOML configuration file |
| `--host <HOST>` | `-H` | `0.0.0.0` | Listen address |
| `--port <PORT>` | `-p` | `50051` | gRPC listen port |
| `--http-port <PORT>` | -- | -- | HTTP Gateway port |
| `--index-dir <PATH>` | -- | `./laurus_index` | Index data directory (global option) |

## Common Configurations

### Development (gRPC only)

```toml
[server]
host = "127.0.0.1"
port = 50051

[index]
data_dir = "./dev_data"
```

```bash
RUST_LOG=debug laurus serve --config config.toml
```

### Production (gRPC + HTTP Gateway)

```toml
[server]
host = "0.0.0.0"
port = 50051
http_port = 8080

[index]
data_dir = "/var/lib/laurus/data"
```

### Minimal (environment variables only)

```bash
export LAURUS_INDEX_DIR=/var/lib/laurus/data
export LAURUS_PORT=50051
export LAURUS_HTTP_PORT=8080
export RUST_LOG=info
laurus serve
```
