# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] - yyyy-mm-dd
 
### Added
- Input arbitrary function in multithreaded algebra
- Test speed of executor closing after each execution of RNG
 
### Changed
 
### Fixed

## \[0.1.1] - 2021-12-22
 
### Changed
- simplified num_threads check, changed alg functions to lowercase as per convention

## \[0.0.7] - 2021-12-22
 
### Added
- Can call random functions directly from mtalg.random
 
### Fixed
- Bug in mtalg/tools

## \[0.0.6] - 2021-12-21

### Changed
- Removed numba from dependencies, added as optional dependency

## \[0.0.5] - 2021-12-14

### Fixed
- np.integer recognized in _check_shape

## \[0.0.4] - 2021-11-26

### Added
 
### Changed
- Functions given simpler names
 
### Fixed

## \[0.0.3] - 2021-11-19

### Added
- Multithreaded algebra with scalars
- Support past standard normal distribution
- Different distributions for random sampling
 
### Changed
 
### Fixed

## \[0.0.2] - 2021-11-19
   
### Added
 
- Generalised multithreaded random number generation
- Benchmarking of multithreaded elementwise algebra
- Benchmarking of multithreaded random number generation 
 
 
## \[0.0.1] - 2021-11-18
   
### Added
 
- Multithreaded elementwise algebra
- Multithreaded random number generation of 2D matrices
