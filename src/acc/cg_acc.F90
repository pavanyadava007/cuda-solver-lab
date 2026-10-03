! Matrix-free CG for the 5-point Poisson problem, Fortran + OpenACC.
!
! Same problem and scaling as include/poisson.hpp. Vectors are stored as
! (0:n+1, 0:n+1) arrays with a zero halo, so the stencil needs no branches.
! The whole solve runs inside one OpenACC data region: only the scalar
! reductions return to the host each iteration.
!
! Built twice by CMake: cg_acc_gpu (gfortran -fopenacc, nvptx offload) and
! cg_acc_host (same source without -fopenacc, i.e. serial Fortran).
!
! Usage: cg_acc_xxx n mode iters tol [csv]
!   mode = fixed (exactly iters iterations) | tol (stop at ||r||/||b|| < tol)
program cg_acc
  use, intrinsic :: iso_fortran_env, only: int64, real64
#ifdef _OPENACC
  use openacc
#endif
  implicit none
  integer, parameter :: dp = real64
  real(dp), allocatable :: x(:,:), r(:,:), p(:,:), ap(:,:), b(:,:)
  real(dp) :: h, rr, rr_new, pap, alpha, beta, bb, err, rres, s, t
  integer :: n, iters, it, i, j, done
  integer(int64) :: c0, c1, rate
  logical :: fixed
  character(len=256) :: arg, csv, mode, impl, device, line
  character(len=8) :: today
  integer(int64) :: nn, nnz
  real(dp) :: tol, ms, model_bytes, gbs, gflops

  call get_command_argument(1, arg); read (arg, *) n
  call get_command_argument(2, mode)
  call get_command_argument(3, arg); read (arg, *) iters
  call get_command_argument(4, arg); read (arg, *) tol
  csv = ''
  if (command_argument_count() >= 5) call get_command_argument(5, csv)
  fixed = (trim(mode) == 'fixed')
#ifdef _OPENACC
  impl = 'fortran_openacc_gfortran'
  call acc_get_property_string(0, acc_device_nvidia, acc_property_name, device)
#else
  impl = 'fortran_serial_gfortran'
  device = 'host_cpu'
#endif

  h = 1.0_dp / (n + 1)
  allocate (x(0:n+1, 0:n+1), r(0:n+1, 0:n+1), p(0:n+1, 0:n+1), ap(0:n+1, 0:n+1), b(0:n+1, 0:n+1))
  x = 0; r = 0; p = 0; ap = 0; b = 0
  do j = 1, n
    do i = 1, n
      b(i, j) = h * h * (mg2(i * h) * g(j * h) + g(i * h) * mg2(j * h))
    end do
  end do
  r = b
  p = b
  bb = sum(b * b)
  rr = bb

  call system_clock(c0, rate)
  done = 0
  !$acc data copyin(r, p) create(ap) copy(x)
  do it = 1, iters
    if (.not. fixed .and. sqrt(rr / bb) < tol) exit

    ! ap = A p and pap = p . ap
    pap = 0
    !$acc parallel loop collapse(2) reduction(+:pap) present(p, ap)
    do j = 1, n
      do i = 1, n
        ap(i, j) = 4 * p(i, j) - p(i-1, j) - p(i+1, j) - p(i, j-1) - p(i, j+1)
        pap = pap + p(i, j) * ap(i, j)
      end do
    end do
    alpha = rr / pap

    ! x += alpha p, r -= alpha ap, rr_new = r . r
    rr_new = 0
    !$acc parallel loop collapse(2) reduction(+:rr_new) present(x, r, p, ap)
    do j = 1, n
      do i = 1, n
        x(i, j) = x(i, j) + alpha * p(i, j)
        r(i, j) = r(i, j) - alpha * ap(i, j)
        rr_new = rr_new + r(i, j) * r(i, j)
      end do
    end do
    beta = rr_new / rr
    rr = rr_new

    ! p = r + beta p
    !$acc parallel loop collapse(2) present(r, p)
    do j = 1, n
      do i = 1, n
        p(i, j) = r(i, j) + beta * p(i, j)
      end do
    end do
    done = it
  end do
  !$acc end data
  call system_clock(c1)
  t = real(c1 - c0, dp) / real(rate, dp)

  ! Checks on the host: true residual and error vs the analytic solution.
  rres = 0
  err = 0
  do j = 1, n
    do i = 1, n
      s = b(i, j) - (4 * x(i, j) - x(i-1, j) - x(i+1, j) - x(i, j-1) - x(i, j+1))
      rres = rres + s * s
      err = max(err, abs(x(i, j) - g(i * h) * g(j * h)))
    end do
  end do
  rres = sqrt(rres / bb)

  ! Same CSV schema as the C++ / CUDA drivers. Traffic model of these three
  ! loops: SpMV + dot 16 N, x/r update + dot 48 N, p update 24 N = 88 N.
  nn = int(n, int64)**2
  nnz = 5_int64 * nn - 4_int64 * n
  ms = 1e3_dp * t / max(done, 1)
  model_bytes = 88.0_dp * real(nn, dp)
  gbs = model_bytes * done / t / 1e9_dp
  gflops = (2.0_dp * real(nnz, dp) + 10.0_dp * real(nn, dp)) * done / t / 1e9_dp
  call date_and_time(date=today)
  ! Numbers first (es/f edit descriptors pad with blanks, squeezed out), then
  ! prepend the text fields, which may legitimately contain blanks.
  write (line, '(i0,a,i0,a,i0,a,a,a,i0,3(a,es13.6),2(a,f0.4),3(a,es10.3),a)') &
    n, ',', nn, ',', nnz, ',0,', trim(mode), ',', done, &
    ',', t, ',', ms, ',', model_bytes, ',', gbs, ',', gflops, &
    ',', rres, ',', err, ',', -1.0_dp, ',3'
  line = today(1:4) // '-' // today(5:6) // '-' // today(7:8) // ',' // trim(device) // ',' // &
         trim(impl) // ',stencil_3loops,' // trim(squeeze(line))
  print '(a)', trim(line)
  if (len_trim(csv) > 0) then
    open (unit=10, file=trim(csv), status='unknown', position='append', action='write')
    write (10, '(a)') trim(line)
    close (10)
  end if
  if (.not. fixed) then
    if (rres < 10 * tol .and. err < 1.1_dp * (2 * 16 * exp(1.0_dp) * 0.4380_dp) / 96 * h * h) then
      print '(a,es10.3,a,es10.3,a,i0)', 'CHECK PASS rel_res_true=', rres, ' max_err=', err, ' iters=', done
    else
      print '(a,es10.3,a,es10.3,a,i0)', 'CHECK FAIL rel_res_true=', rres, ' max_err=', err, ' iters=', done
      stop 1
    end if
  end if
contains

  ! Analytic solution u = g(x) g(y), g(t) = t (1 - t) e^t; mg2 = -g''.
  pure real(dp) function g(t)
    real(dp), intent(in) :: t
    g = t * (1 - t) * exp(t)
  end function g

  pure real(dp) function mg2(t)
    real(dp), intent(in) :: t
    mg2 = t * (t + 3) * exp(t)
  end function mg2

  ! List-directed es/f output pads with blanks; CSV fields must not contain them.
  function squeeze(str) result(out)
    character(len=*), intent(in) :: str
    character(len=len(str)) :: out
    integer :: k, m
    out = ' '
    m = 0
    do k = 1, len_trim(str)
      if (str(k:k) /= ' ') then
        m = m + 1
        out(m:m) = str(k:k)
      end if
    end do
  end function squeeze
end program cg_acc
