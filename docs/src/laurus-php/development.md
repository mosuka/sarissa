# Development Setup

This page covers how to set up a local development environment
for the `laurus-php` binding, build it, and run the test
suite.

## Prerequisites

- **Rust** 1.85 or later with Cargo
- **PHP** 8.1 or later with development headers (`php-dev` / `php-devel`)
- **Composer** for dependency management
- Repository cloned locally

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
```

## Build

### Development build

Compiles the Rust native extension in debug mode. Re-run after
any Rust source change.

```bash
cd laurus-php
cargo build
```

The resulting shared library is located at `../target/debug/liblaurus_php.so`.

### Release build

```bash
cd laurus-php
cargo build --release
```

The resulting shared library is located at `../target/release/liblaurus_php.so`.

### Verify the build

```bash
php -d extension=../target/release/liblaurus_php.so -r "
use Laurus\Index;
\$index = new Index();
print_r(\$index->stats());
"
# Array ( [documentCount] => 0 [vectorFields] => Array ( ) )
```

## Testing

Tests use [PHPUnit](https://phpunit.de/) and are located in
`tests/`. Composer is used only for these dev-time PHP dependencies —
the runtime extension itself is built and loaded directly by Cargo.

```bash
# Install test dependencies (PHPUnit only)
composer install

# Run all tests
php -d extension=../target/release/liblaurus_php.so vendor/bin/phpunit tests/
```

To run a specific test file:

```bash
php -d extension=../target/release/liblaurus_php.so vendor/bin/phpunit tests/LaurusTest.php
```

## Linting and formatting

```bash
# Rust lint (Clippy)
cargo clippy -p laurus-php -- -D warnings

# Rust formatting
cargo fmt -p laurus-php --check

# Apply formatting
cargo fmt -p laurus-php
```

## Cleaning up

```bash
# Remove build artifacts
cargo clean

# Remove Composer dependencies
rm -rf vendor/
```

## Workspace integration and clang-sys

`laurus-php` uses [ext-php-rs](https://github.com/extphprs/ext-php-rs), whose
bindings generator (`ext-php-rs-bindgen`) parses PHP headers with `bindgen`,
which loads `libclang` through `clang-sys`. The `laurus-ruby` crate depends on
`magnus`, whose `rb-sys` build also uses `bindgen` + `clang-sys`. Cargo
forbids two packages with the same `links` value in a single workspace, so
both PHP and Ruby bindings can only coexist as workspace members while they
resolve to the *same* `clang-sys` package.

Older `ext-php-rs` releases depended on `ext-php-rs-clang-sys` (a fork of
`clang-sys` that also declared `links = "clang"`), which conflicted with the
original `clang-sys` pulled in by `laurus-ruby`. That era required a local
`[patch.crates-io]` override (a vendored copy under `patches/` with the
`links` declaration removed). Since `ext-php-rs-bindgen 0.72.1-extphprs.2`
(pulled in by `ext-php-rs` 0.15.15), the fork is gone — `ext-php-rs` depends
on the regular `clang-sys` again, both bindings share one `clang-sys`
package, and the patch has been removed.

If a future `ext-php-rs` upgrade reintroduces a forked `clang-sys` (watch for
a `links = "clang"` conflict error from `cargo build -p laurus-php -p
laurus-ruby`), reintroduce the vendored patch: copy the forked crate's source
into `patches/`, comment out its `links = "clang"` line, and point
`[patch.crates-io]` in the root `Cargo.toml` at it. This is safe because
`clang-sys` uses `libclang` only at build time (for `bindgen` header parsing)
and does not link it into the final binary.

## macOS linker flag (`-undefined dynamic_lookup`)

PHP extensions are shared libraries (`.so` / `.dylib`) that are loaded by the
PHP interpreter at runtime. They reference PHP API symbols (`zend_*`,
`php_*`, etc.) that are defined in the PHP binary itself, not in any library
the extension links against. On Linux the linker allows undefined symbols in
shared libraries by default, so this works without extra flags. On macOS the
linker treats undefined symbols as errors, which causes the build to fail:

```text
ld: symbol(s) not found for architecture arm64
```

The fix is to pass `-Wl,-undefined,dynamic_lookup` to the linker, which
tells it to defer symbol resolution to load time (when PHP `dlopen`s the
extension).

This flag is **not** set in `.cargo/config.toml` because it would apply to
every crate in the workspace, including non-PHP crates where undefined
symbols should remain errors. Instead it is applied only when building
`laurus-php`:

**Makefile** (local development):

```makefile
build-laurus-php:
ifeq ($(shell uname -s),Darwin)
    RUSTFLAGS="-C link-args=-Wl,-undefined,dynamic_lookup" cargo build -p laurus-php --release
else
    cargo build -p laurus-php --release
endif
```

**CI** (GitHub Actions):

```yaml
- name: Build PHP extension
  shell: bash
  run: |
    if [ "$RUNNER_OS" == "macOS" ]; then
      export RUSTFLAGS="-C link-args=-Wl,-undefined,dynamic_lookup"
    fi
    cargo build --release -p laurus-php
```

When building on macOS, always use `make build-laurus-php` or
`make test-laurus-php` instead of running `cargo build -p laurus-php`
directly.

## Project layout

```text
laurus-php/
├── Cargo.toml          # Rust crate manifest
├── composer.json       # Composer package definition
├── composer.lock       # Locked dependency versions
├── src/                # Rust source (ext-php-rs binding)
│   ├── lib.rs          # Module registration
│   ├── index.rs        # Index class
│   ├── schema.rs       # Schema class
│   ├── query.rs        # Query classes
│   ├── search.rs       # SearchRequest / SearchResult / Fusion
│   ├── analysis.rs     # Tokenizer / Filter / Token
│   ├── convert.rs      # PHP <-> DataValue conversion
│   └── errors.rs       # Error mapping
├── tests/              # PHPUnit tests
│   └── LaurusTest.php
└── examples/           # Runnable PHP examples
```
