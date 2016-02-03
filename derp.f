C     TODO:
C
C     + Analytic gradient.
C     + Back propagation.      
C     + Back propagation
C     + Batch gradient descent.
      
      program derp
        
      parameter ( nx=2, nh=2, ny=1, ntrain=4 )
      real x(nx), h(nh), y(ny)
      real hw(nh,nx), yw(ny,nh)
      real hg(nh,nx), yg(ny,nh)
      real xtrain(nx,ntrain), ytrain(ny,ntrain)
      
C     Setup training examples      
      
      data xtrain / 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 /
      data ytrain / 0.0, 1.0, 1.0, 0.0 /
      
      call random_number(hw)
      call random_number(yw)

      call cost(ecost)
      
      iter = 0
      
      do while ( ecost > 0.1 )
        print*, 'cost', ecost
        print*, 'iter', iter
        iter = iter + 1
      end do

      contains
        
      elemental function sigmoid ( x )
        intent(in) x
        sigmoid = 1.0 / ( 1.0 + exp(-x) )
      end
      
      elemental function sigmoid_prime ( x )
        intent(in) x
        s = sigmoid(x)
        sigmoid_prime = s * ( 1.0 - s)
      end
        
      subroutine forward
        h = matmul(hw,x)
        h = sigmoid(h)  
        y = matmul(yw,h)
        y = sigmoid(y)
      end
      
      subroutine analytic_gradient
        delta(1,3) = y(1) - ytrain(
C       real hg(nh,nx), yg(ny,nh)
        yg(1,1) = delta(1,3) * sigmoid_prime(
      end

      subroutine backward 
      end
      
      subroutine stocastic
      end

      subroutine batch
      end
      
      subroutine cost ( ecost )
        ecost = 0.0
        do k = 1, ntrain
          x = xtrain(:,k)
          call forward
          ecost = ecost + 0.5 * sum((y-ytrain(:,k))**2)
        end do
      end
        
      end program derp
