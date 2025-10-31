# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

You should also add project tags for each release in Github, see [Managing releases in a repository](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

## [Unreleased]
### Added
- Script for evaluating models against the Buckeye test dataset
- Draw training samples from Buckeye data matching certain speaker demographics or individuals
- Unit tests with pytest
- Buckeye Corpus support in preprocessing and model training
- Resources directory for pronounciation dictionary and IPA vocabulary files
- Bash scripts for installation and training on slurm added in the `scripts` folder
- GitHub action to run tests
- `visualization` module to support creating standardized plots across jupyter notebooks

### Changed
- All build and packaging switched to use only pyproject.toml
- Scripts read in files from user-specified paths rather than hard coded paths
- Separated Librispeech (English) data processing and model training from Common Voice (other languages)
- Corpus-specific vocabulary files are loaded to compare expected vocabulary against actual vocabulary from training data
- Added support for corpora with whitespace separated phonemes to determine vocabulary

### Fixed
- Drop rows with null values from Buckeye transcriptions
- Drop duplicate rows in Buckeye transcriptions
- Fixed bug in building vocabulary
- Fixed bug in English to IPA conversion where keep_suprasegmental flag was ignored

### Removed
