# Changelog

## [0.2.1](https://github.com/thyn-ai/codna/compare/cli-v0.2.0...cli-v0.2.1) (2026-08-18)


### Bug Fixes

* **publish:** remove the duplicate release-triggered publish path ([#467](https://github.com/thyn-ai/codna/issues/467)) ([cc161db](https://github.com/thyn-ai/codna/commit/cc161dbacc9ee967a5b2979b570e55d82231746c))

## [0.2.0](https://github.com/thyn-ai/codna/compare/cli-v0.1.49...cli-v0.2.0) (2026-08-17)


### Features

* **packaging:** stage memengine from the signed telys bundle (option-B slice 1) ([#460](https://github.com/thyn-ai/codna/issues/460)) ([dfe2c4a](https://github.com/thyn-ai/codna/commit/dfe2c4a38045850e1cbb6205008cf8528a2fefaa))
* **publish:** wheel memengine comes from the signed bundle, never the repo (option-B slice 2) ([#461](https://github.com/thyn-ai/codna/issues/461)) ([ac2cb22](https://github.com/thyn-ai/codna/commit/ac2cb2208902622ffdf697c82f7da88c357112bb))
* **webhook:** consistently wire the org's own BYOK provider key into every fix ([#462](https://github.com/thyn-ai/codna/issues/462)) ([ad299da](https://github.com/thyn-ai/codna/commit/ad299da373070c2fd5ab1c539b7736e94277f3a9))


### Bug Fixes

* **testrun:** detect pytest config one level down (monorepo-aware) ([#463](https://github.com/thyn-ai/codna/issues/463)) ([ce2f7cc](https://github.com/thyn-ai/codna/commit/ce2f7cceac49917ad17c6f8b83755320e1169d91))

## [0.1.49](https://github.com/thyn-ai/codna/compare/cli-v0.1.48...cli-v0.1.49) (2026-08-17)


### Bug Fixes

* **webhook:** a linked org's codna-fix label / CI fix always crashed ([#451](https://github.com/thyn-ai/codna/issues/451)) ([544fff2](https://github.com/thyn-ai/codna/commit/544fff2b776a395c434a3a033a1b72a165f83205))


### Performance Improvements

* **codeunits:** split source once per file — Python extraction 5.0x faster ([#456](https://github.com/thyn-ai/codna/issues/456)) ([36316cd](https://github.com/thyn-ai/codna/commit/36316cde37c844d1004145bd3bce5b4e12f7800d))
