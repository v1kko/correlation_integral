! Correlation Integral caluclations done fast with openmp
! Copyright (C) 2021  Victor Azizi - victor@lipsum.eu

! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.

subroutine manhattan(data, dims, r, cd)
use iso_fortran_env
implicit none
real(real32)  , intent(in) :: data(:)
integer       , intent(in) :: dims
real(real32)  , intent(in) :: r(:)
real(real32)  , intent(out):: cd(size(r))

integer(int64)             :: cd_c(size(r))
integer(int64), allocatable:: cd_p(:,:)
integer                    :: L, L_2, i, j, k, datasize, rsize
real(real32)               :: dist

cd_c = 0

datasize = size(data)
rsize = size(r)
L = datasize - dims
L_2 = L/2  

allocate(cd_p(rsize,L))
cd_p = 0

!$OMP PARALLEL DO PRIVATE(dist) schedule(guided)
do i=L,1,-1
  do j=i+1,L
    dist = 0
    do k=1,dims
      dist = dist + abs ( data(i+k) - data(j+k) )
    enddo
    do k=1,rsize
      if (r(k) > dist) then
        cd_p(k,i) = cd_p(k,i) + 1
      endif
    enddo
  enddo
enddo
!$OMP END PARALLEL DO

cd_c = sum( cd_p,2)
cd = real(real(cd_c, real64) / (real(L, real64) * (real(L, real64) - 1. ) / 2.),real32)

deallocate(cd_p)
end subroutine

subroutine euclidean(data, dims, r, cd)
use iso_fortran_env
implicit none
real(real32)  , intent(in) :: data(:)
integer       , intent(in) :: dims
real(real32)  , intent(in) :: r(:)
real(real32)  , intent(out):: cd(size(r))

integer(int64)             :: cd_c(size(r))
integer(int64), allocatable:: cd_p(:,:)
integer                    :: L, L_2, i, j, k, datasize, rsize
real(real32)               :: dist

cd_c = 0

datasize = size(data)
rsize = size(r)
L = datasize - dims
L_2 = L/2  

allocate(cd_p(rsize,L))
cd_p = 0

!$OMP PARALLEL DO PRIVATE(dist) schedule(guided)
do i=L,1,-1
  do j=i+1,L
    dist = 0
    do k=1,dims
      dist = dist + ( data(i+k) - data(j+k) )**2
    enddo
    do k=1,rsize
      dist = dist**0.5
      if (r(k) > dist) then
        cd_p(k,i) = cd_p(k,i) + 1
      endif
    enddo
  enddo
enddo
!$OMP END PARALLEL DO

cd_c = sum( cd_p,2)
cd = real(real(cd_c, real64) / (real(L, real64) * (real(L, real64) - 1. ) / 2.),real32)

deallocate(cd_p)
end subroutine

subroutine chebyshev(data, dims, r, cd)
use iso_fortran_env
implicit none
real(real32)  , intent(in) :: data(:)
integer       , intent(in) :: dims
real(real32)  , intent(in) :: r(:)
real(real32)  , intent(out):: cd(size(r))

integer(int64)             :: cd_c(size(r))
integer(int64), allocatable:: cd_p(:,:)
integer                    :: L, L_2, i, j, k, datasize, rsize
real(real32)               :: dist

cd_c = 0

datasize = size(data)
rsize = size(r)
L = datasize - dims
L_2 = L/2  

allocate(cd_p(rsize,L))
cd_p = 0

!$OMP PARALLEL DO PRIVATE(dist) schedule(guided)
do i=L,1,-1
  do j=i+1,L
    dist = 0
    do k=1,dims
      dist = max(dist, abs(data(i+k) - data(j+k)))
    enddo
    do k=1,rsize
      if (r(k) > dist) then
        cd_p(k,i) = cd_p(k,i) + 1
      endif
    enddo
  enddo
enddo
!$OMP END PARALLEL DO

cd_c = sum( cd_p,2)
cd = real(real(cd_c, real64) / (real(L, real64) * (real(L, real64) - 1. ) / 2.),real32)

deallocate(cd_p)
end subroutine
