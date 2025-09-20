!-------------------------------------------------------------------------------
!> @brief global_grid module
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/10/2020
!
!  REVISION HISTORY:
!  01/10/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_mod

    use global_grid_data_mod, only: global_grid_type
    use global_grid_base_mod, only: global_grid_init, global_grid_end

    implicit none
    private

    public :: global_grid_type
    public :: global_grid_init, global_grid_end

end module global_grid_mod

