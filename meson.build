project(
  'numba_nnls',
  'fortran',
  version: '0.1.0',
  default_options: [
    'buildtype=release',
  ]
  )

py3 = import('python').find_installation(pure: false)

shared_library('nnlsnumba',
    [
    'src/nnls.f', 
    'src/nnls_capi.f90'
    ],
    install: true,
    install_dir: py3.get_install_dir() / 'numba_nnls',
)

subdir('numba_nnls')
