C     Naive implementation of backpropagation  to aid understanding of the principles.

      program derp
      implicit none

      integer nx, nh, ny, ntrain
      parameter ( nx=2, nh=2, ny=1, ntrain=4 )
      real        x(nx), h(nh), y(ny)

C     a(i,j): weight from node x(j) to node h(i).
C     b(i,j): weight from node h(j) to node y(i).
C     same convention as bprod.m in NNSYMID package (minus bias unit):
C     http://www.iau.dtu.dk/research/control/nnsysid.html

      real a(nh,nx), b(ny,nh)
      real agrad(size(a,1),size(a,2)), bgrad(size(b,1),size(b,2))
      real ydelta(ny), hdelta(nh)
      real xtrain(nx,ntrain), ytrain(ny,ntrain)

C     Setup training examples

      data xtrain / 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 /
      data ytrain / 0.0, 1.0, 1.0, 0.0 /

      integer i, j, iter, niter
      real    ecost
      real    tdelta, t0, t1
      real    alpha
      parameter ( niter = 100000 )

C     Initialize weights to something random.

      print*,'=================================================='

      print*, 'Initializing random weights.'

      call random_number(a)
      call random_number(b)

      print*,'=================================================='

      call cost(ecost)

      call printmat('cost', reshape([ecost], [1,1]))

      print*,'=================================================='

      print*, 'Numerical gradient with central differences.'

      agrad = 0.0
      bgrad = 0.0

      call ngrad()

      call printmat('dE/da', agrad)
      call printmat('dE/db', bgrad)

      print*,'=================================================='

      print*, 'Gradient with backpropagation.'

      agrad = 0.0
      bgrad = 0.0

      call bpgrad()

      call printmat('dE/da', agrad)
      call printmat('dE/db', bgrad)

      print*,'=================================================='

      print*, 'Train a xor function to a feed forward neural network'

      call random_number(a)
      call random_number(b)

      alpha = 2.0
      iter = 1

      do
        call cost(ecost)
C       call bpgrad()
        call ngrad()
        a = a - alpha*agrad
        b = b - alpha*bgrad

        if (mod(iter, 1000) == 0) print*, 'iter:', iter,
     $                                    'cost:', ecost
        if (ecost < 1e-3) exit
        iter = iter + 1
      end do
      print*, 'iter:', iter, 'cost:', ecost

      print*,'=================================================='

      do i = 1, ntrain
        x = xtrain(:,i)
        call forward
        call printmat('x', reshape(x,[1,nx]))
        call printmat('y', reshape([y],[1,1]))
      end do

      print*,'=================================================='

      call cpu_time(t0)

      do iter = 1, niter
        call ngrad()
      end do

      call cpu_time(t1)
      tdelta = t1 - t0
      print*, 'Central differences ', niter, ' iterations:', tdelta

      call cpu_time(t0)

      do iter = 1, niter
        call bpgrad()
      end do

      call cpu_time(t1)
      tdelta = t1 - t0
      print*, 'Backpropagation ', niter, ' iterations:', tdelta

      print*,'=================================================='

      contains

      subroutine printmat ( name, smat )
        character(len=*)  name
        real              smat(:,:)
        integer           i
        print*, trim(name), ':'
        do i = 1, size(smat,1)
          print*, smat(i,:)
        end do
      end

      real elemental function sigmoid ( x )
        real, intent(in) :: x
        sigmoid = 1.0 / ( 1.0 + exp(-x) )
      end

      subroutine bnorm ( x )
        real        x(:), xmean, xvar, eps
        integer     m
        parameter ( eps = 100.0 * epsilon(1.0) )
        m = size(x)
        xmean = sum(x) / m
        xvar = sum( ( x - xmean )**2 ) / m
        x = ( x - xmean ) / sqrt(xvar + eps)
        x = 0.8*sqrt(xvar)*x + 3.0*xmean
      end

      subroutine forward ()
        h = matmul(a,x)
        call bnorm(h)
        h = sigmoid(h)
        y = matmul(b,h)
        call bnorm(y)
        y = sigmoid(y)
      end

      subroutine cost ( ecost )
        real              ecost
        integer           k
        ecost = 0.0
        do k = 1, ntrain
          x = xtrain(:,k)
          call forward
          ecost = ecost + (0.5 / ntrain) * sum((y-ytrain(:,k))**2)
        end do
      end

C     compute a numerical approximation of the gradient

      subroutine ngrad ()
        real              delta, aij, bij, ep, en
        integer           i, j
        parameter       ( delta = sqrt(epsilon(1.0)) )
C       Approximate dE/da with central differences.
        do j = 1, size(a,2)
          do i = 1, size(a,1)
            aij = a(i,j)
            a(i,j) = aij + delta
            call cost(ep)
            a(i,j) = aij - delta
            call cost(en)
            agrad(i,j) = (ep-en)/(2*delta)
            a(i,j) = aij
          end do
        end do
C       Approximate dE/db with central differences.
        do j = 1, size(b,2)
          do i = 1, size(b,1)
            bij = b(i,j)
            b(i,j) = bij + delta
            call cost(ep)
            b(i,j) = bij - delta
            call cost(en)
            bgrad(i,j) = (ep-en)/(2*delta)
            b(i,j) = bij
          end do
        end do
      end

C     compute gradient with backpropagation NO UPDATE

      subroutine bpgrad ()
        integer           i, j, k
        agrad = 0.0
        bgrad = 0.0
        do k = 1, ntrain
          x = xtrain(:,k)
          call forward
          do i = 1, ny
            ydelta(i) = ( y(i) - ytrain(i,k) ) * y(i) * ( 1.0 - y(i) )
     $                * (1.0 / ntrain)
          end do
C     compute dE/db
          do j = 1, size(b,2)
            do i = 1, size(b,1)
              bgrad(i,j) = bgrad(i,j) + ydelta(i) * h(j)
            end do
          end do
C     compute hdelta
          hdelta = 0.0
          do j = 1, nh
            hdelta(j) = h(j)*(1.0-h(j))*dot_product(ydelta,b(:,j))
C           hdelta(j) = 0.0
C           do i = 1, ny
C             hdelta(j) = hdelta(j)
C    $                  + ydelta(i) * h(j) * ( 1.0 - h(j) ) * b(i,j)
C           end do
          end do
C     compute dE/da
C       This is a rank-1 update of agrad. Could be optimized with BLAS.
          do j = 1, size(a,2)
            do i = 1, size(a,1)
              agrad(i,j) = agrad(i,j) + hdelta(i) * x(j) ! <<< this is right
            end do
          end do
C     end compute dE/da
        end do
      end

      subroutine stocastic
      end

      subroutine batch
      end

      end program derp
