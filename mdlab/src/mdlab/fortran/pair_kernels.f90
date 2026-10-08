module pair_kernels

   !=========================================================================!
   !                                                                         !
   !  LENNARD-JONES ENERGY AND FORCES OVER A PAIR LIST                       !
   !                                                                         !
   !=========================================================================!
   !                                                                         !
   !  PURPOSE                                                                !
   !  -------                                                                !
   !  The inner loop of a pair potential, written for speed and called from  !
   !  Python through f2py. Each listed pair is reduced to its nearest image  !
   !  in a rectangular box, tested against the cutoff, and its energy and    !
   !  forces added in. The energy is shifted to zero at the cutoff.          !
   !                                                                         !
   !  CONVENTIONS                                                            !
   !  -----------                                                            !
   !  * positions(3, n_atoms): column a is atom a, as NumPy's (n_atoms, 3)   !
   !    array reads when passed transposed, so no copy is made               !
   !  * pair_i, pair_j: indices counted from 0, as Python counts them        !
   !  * any consistent units: the energy in those of well_depth, lengths in  !
   !    those of sigma and the box                                           !
   !                                                                         !
   !  AUTHOR                                                                 !
   !  ------                                                                 !
   !  BCA - October 2026                                                     !
   !                                                                         !
   !=========================================================================!

   implicit none

   integer, parameter :: dp = selected_real_kind(15, 300)

contains

   subroutine lj_energy_forces(n_atoms, n_pairs, positions, box, pair_i,     &
                               pair_j, well_depth, sigma, r_cut, energy,     &
                               forces)

      !======================================================================!
      !  SUM OVER THE PAIRS                                                  !
      !  ------------------                                                  !
      !  For each pair, d = r_j - r_i is reduced by whole boxes and, inside  !
      !  the cutoff, phi(r) = 4 eps (s^12 - s^6) - phi(r_cut) with s = sig/r !
      !  is added. The push 24 eps (2 s^12 - s^6) / r^2 times d is the force !
      !  on j; the force on i is its opposite.                               !
      !======================================================================!

      implicit none

      !----------------------------------------------------------------------!
      !  Arguments                                                           !
      !----------------------------------------------------------------------!

      integer,       intent(in)  :: n_atoms
      integer,       intent(in)  :: n_pairs
      real(kind=dp), intent(in)  :: positions(3, n_atoms)
      real(kind=dp), intent(in)  :: box(3)
      integer,       intent(in)  :: pair_i(n_pairs)
      integer,       intent(in)  :: pair_j(n_pairs)
      real(kind=dp), intent(in)  :: well_depth
      real(kind=dp), intent(in)  :: sigma
      real(kind=dp), intent(in)  :: r_cut
      real(kind=dp), intent(out) :: energy
      real(kind=dp), intent(out) :: forces(3, n_atoms)

      !----------------------------------------------------------------------!
      !  Pair Quantities                                                     !
      !                                                                      !
      !  sep     : separation r_j - r_i, then its nearest image              !
      !  r2      : squared length of sep                                     !
      !  s6      : (sigma^2 / r2)^3                                          !
      !  push    : 24 eps (2 s^12 - s^6) / r^2, the force per unit of sep    !
      !  phi_cut : the unshifted energy at the cutoff                        !
      !----------------------------------------------------------------------!

      real(kind=dp) :: sep(3)
      real(kind=dp) :: r2
      real(kind=dp) :: s6
      real(kind=dp) :: push
      real(kind=dp) :: phi_cut
      real(kind=dp) :: sigma2
      real(kind=dp) :: r_cut2
      integer       :: p
      integer       :: i
      integer       :: j

      sigma2  = sigma * sigma
      r_cut2  = r_cut * r_cut
      s6      = (sigma2 / r_cut2)**3
      phi_cut = 4.0_dp * well_depth * (s6 * s6 - s6)
      energy  = 0.0_dp
      forces  = 0.0_dp

      do p = 1, n_pairs
         i   = pair_i(p) + 1
         j   = pair_j(p) + 1
         sep = positions(:, j) - positions(:, i)
         sep = sep - box * anint(sep / box)
         r2  = sum(sep * sep)
         if (r2 >= r_cut2) cycle
         s6           = (sigma2 / r2)**3
         push         = 24.0_dp * well_depth * (2.0_dp * s6 * s6 - s6) / r2
         energy       = energy + 4.0_dp * well_depth * (s6 * s6 - s6)        &
                        - phi_cut
         forces(:, i) = forces(:, i) - push * sep
         forces(:, j) = forces(:, j) + push * sep
      end do

   end subroutine lj_energy_forces

end module pair_kernels
