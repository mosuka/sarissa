# Installation

## Prebuilt binaries

Each [GitHub release](https://github.com/mosuka/laurus/releases) ships
prebuilt `laurus` binaries for the following targets, all built with
`--features embeddings-all`:

| Target triple | Platform | libc | Linking | Archive |
| :--- | :--- | :--- | :--- | :--- |
| `x86_64-unknown-linux-gnu` | Linux x86_64 | glibc | dynamic | `.tar.gz` |
| `aarch64-unknown-linux-gnu` | Linux aarch64 | glibc | dynamic | `.tar.gz` |
| `x86_64-unknown-linux-musl` | Linux x86_64 | musl | static | `.tar.gz` |
| `aarch64-unknown-linux-musl` | Linux aarch64 | musl | static | `.tar.gz` |
| `x86_64-apple-darwin` | macOS Intel | -- | dynamic | `.tar.gz` |
| `aarch64-apple-darwin` | macOS Apple Silicon | -- | dynamic | `.tar.gz` |
| `x86_64-pc-windows-msvc` | Windows x86_64 | MSVC | dynamic | `.zip` |
| `aarch64-pc-windows-msvc` | Windows arm64 | MSVC | dynamic | `.zip` |

```bash
VERSION=v0.13.1
TARGET=x86_64-unknown-linux-musl
curl -fsSL -O "https://github.com/mosuka/laurus/releases/download/${VERSION}/laurus-${VERSION}-${TARGET}.tar.gz"
tar -xzf "laurus-${VERSION}-${TARGET}.tar.gz"
./laurus --version
```

### Which build should I use?

- Use a **gnu** build on an ordinary Linux distribution. musl's allocator
  (`mallocng`) is measurably slower than glibc's under multi-threaded,
  allocation-heavy workloads, which describes laurus's indexing path.
- Use a **musl** build for Alpine, distroless, `scratch`, or any host whose
  glibc is older than the build runner's -- the musl binaries are fully
  statically linked and have no dynamic library dependencies at all.

### Using the musl binary in a container

```dockerfile
FROM alpine:3.22
# Only needed if you use `embeddings-openai`: reqwest's rustls backend
# verifies against the OS trust store. Hugging Face model downloads
# (`embeddings-candle` / `embeddings-multimodal`) use root certificates
# bundled into the binary and work without this. See "Feature Flags" in
# the development guide for details.
RUN apk add --no-cache ca-certificates
COPY laurus /usr/local/bin/laurus
ENTRYPOINT ["laurus"]
```

or, with no package manager at all:

```dockerfile
FROM scratch
COPY laurus /laurus
ENTRYPOINT ["/laurus"]
```

A couple of runtime notes for musl/`scratch` deployments:

- DNS resolution uses musl's built-in resolver, which reads
  `/etc/resolv.conf` and ignores NSS plugins. Container runtimes always
  provide `/etc/resolv.conf`, so this works out of the box; a hand-built
  `scratch` image must not omit it.
- Rust's standard library uses its own default stack size for spawned
  threads (2 MiB), not musl's smaller `pthread` default, so this is not
  usually a concern in practice.

## From crates.io

```bash
cargo install laurus-cli
```

This installs the `laurus` binary to `~/.cargo/bin/`.

## From source

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
cargo install --path laurus-cli
```

## Verify

```bash
laurus --version
```
