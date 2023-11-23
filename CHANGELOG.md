# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## \[Unreleased] - yyyy-mm-dd
 
### Added

## \[1.1.1] - 2023-11-23
 
### Added
- Sphinx docs

### Changed
- Removed CDSW threads check

## \[1.1.0] - 2022-01-05
 
### Changed
- cpu_count() as default for _global_num_threads
  
### Added
 
- Added `get_num_threads`

### Fixed

- Improved documentation

## \[1.0.0] - 2021-12-28
 
### Added
- integer and random methods to MRNG
- set_num_threads function
- different generators can be chosen for MRNG
- return of float when size=None for MRNG
- all MRNG methods can be accessed from mtalg.random as well
- More documentation

### Changed
- MRNG methods returns instead of haveing to access mrng.values
- MRNG attributes made private

### Fixed
- Handling of scalars in alg functions

## \[0.1.4] - 2021-12-24
 
### Added
- Almost all distributions

## \[0.1.2, 0.1.3] - 2021-12-23
 
### Fixed
- Github pipeline

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
