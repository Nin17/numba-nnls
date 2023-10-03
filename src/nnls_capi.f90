module nnls_capi
    use iso_c_binding, only : c_double, c_int
    implicit none
    private
    public nnls_c, nnls_c_wrapper

    contains
        subroutine NNLS_C_WRAPPER(A, MDA, M, N, B, X, RNORM, W, ZZ, INDEX, MODE, &
            &MAXITER) bind(c, name="nnls_c_wrapper")
            implicit none
            
            integer(c_int), intent(in) :: MDA
            integer(c_int), intent(in) :: M
            integer(c_int), intent(in) :: N
            integer(c_int), intent(in) :: INDEX(N)
            integer(c_int), intent(in) :: MODE
            integer(c_int), intent(in) :: MAXITER
            real(c_double), intent(inout) :: A(MDA, N)
            real(c_double), intent(inout) :: B(M)
            real(c_double), intent(inout) :: X(N)
            real(c_double), intent(in) :: RNORM
            real(c_double), intent(in) :: W(N)
            real(c_double), intent(in) :: ZZ(M)

            call NNLS(A, M, M, N, B, X, RNORM, W, ZZ, INDEX, MODE, MAXITER)
            
        end subroutine nnls_c_wrapper


        subroutine NNLS_C(A, M, N, B, X, RNORM, MODE, MAXITER) bind(c, name="nnls_c")
            implicit none

            integer(c_int), intent(in) :: M
            integer(c_int), intent(in) :: N
            integer(c_int), intent(in) :: MODE
            integer(c_int), intent(in) :: MAXITER
            real(c_double), intent(inout) :: A(M, N)
            real(c_double), intent(inout) :: B(M)
            real(c_double), intent(out) :: X(N)
            real(c_double), intent(in) :: RNORM


            real(c_double), allocatable :: W(:)
            real(c_double), allocatable :: ZZ(:)
            integer(c_int), allocatable :: INDEX(:)
            
            allocate (W(N))
            allocate (ZZ(M))
            allocate (INDEX(N))

            call NNLS(A, M, M, N, B, X, RNORM, W, ZZ, INDEX, MODE, MAXITER)

            deallocate (W)
            deallocate (ZZ)
            deallocate (INDEX)
        end subroutine nnls_c
end module nnls_capi