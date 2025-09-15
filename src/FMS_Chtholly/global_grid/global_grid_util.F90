!-------------------------------------------------------------------------------
!> @brief util subroutines for global_grid_mod
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/21/2020
!
!  REVISION HISTORY:
!  01/21/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_util_mod

    !------ fms modules
    !------ AC modules
    use lib_grid_mod,           only: R_GRID

    implicit none
    private

    public :: global_grid_update_kik

contains
    !===========================================================================
    subroutine global_grid_update_kik(pt_ext, pt_kik, isd, ied, ng, np)
        integer, intent(in) :: isd, ied, ng, np
        real(kind=R_GRID), dimension(np,isd:ied,isd:ied,6), intent(in) :: pt_ext
        real(kind=R_GRID), dimension(np,isd:ied,isd:ied,6), intent(out):: pt_kik 
        !--- local
        integer :: is, ie, js, je, jsd, jed
        integer :: i, j, k, n, n_src, ii, jj
        integer :: nw, ne, ns, nn, n_neighbor(4)
        integer :: isk, iek, jsk, jek

        !--- assign dims
        jsd = isd
        jed = ied

        is = isd+ng
        js = jsd+ng
        ie = ied-ng
        je = jed-ng

        !--- copy inner values
        pt_kik(:,:,:,:) = -999.
        do n = 1, 6
            do j = js, je
                do i = is, ie

                    pt_kik(:, i, j, n) = pt_ext(:, i, j, n)

                enddo
            enddo
        enddo

        do n = 1,6

            call get_neighbor_tile_num(n, nw, ne, ns, nn)
            n_neighbor = [nw,ne,ns,nn]
            do k = 1,4
                n_src = n_neighbor(k)

                call get_neighbor_bounds(isk,iek,jsk,jek,n,n_src,ie,je,ng)
                do j = jsk,jek
                    do i = isk,iek
                        call get_neighbor_index(ii,jj,i,j,n,n_src,ie,je)
                        pt_kik(:,i,j,n) = pt_kik(:,ii,jj,n_src)
                    enddo
                enddo

            enddo

        enddo

    end subroutine global_grid_update_kik
    !=============================================================================
    subroutine get_neighbor_index(ii,jj,i,j,n,n_src,npx,npy)
        integer, intent(out) :: ii,jj
        integer, intent(in) :: i,j,n,n_src,npx,npy
        ! local
        integer :: nw, ne, ns, nn, isc=0, jsc=0, iec=0, jec=0
        logical :: is_even_tile_num

        is_even_tile_num = mod(n,2).eq.0

        isc = 1; jsc = 1; iec = npx; jec = npy
        call get_neighbor_tile_num(n,nw,ne,ns,nn)

        if (.not.is_even_tile_num) then
            if (n_src.eq.nw) then
                ii = iec-(j-jsc)
                jj = jec+(i-isc)
            endif
            if (n_src.eq.ne) then
                ii = isc+(i-iec)
                jj = jsc+(j-jsc)
            endif
            if (n_src.eq.ns) then
                ii = isc+(i-isc)
                jj = jec+(j-jsc)
            endif
            if (n_src.eq.nn) then
                ii = isc+(j-jec)
                jj = jec-(i-isc)
            endif
        else  ! is_even_tile_num
            if (n_src.eq.nw) then
                ii = iec+(i-isc)
                jj = jsc+(j-jsc)
            endif
            if (n_src.eq.ne) then
                ii = iec-(j-jsc)
                jj = jsc+(i-iec)
            endif
            if (n_src.eq.ns) then
                ii = iec+(j-jsc)
                jj = jec-(i-isc)
            endif
            if (n_src.eq.nn) then
                ii = isc+(i-isc)
                jj = jsc+(j-jec)
            endif
        endif ! is_even_tile_num

    end subroutine get_neighbor_index
    !===========================================================================
    subroutine get_neighbor_bounds(is,ie,js,je,n,n_src,npx,npy,ng)
        ! n refer to the tile to get kik edge assigned
        ! n_src refer to the neighbor tile of the source pt info
        integer, intent(out) :: is,ie,js,je
        integer, intent(in) :: n,n_src,npx,npy,ng
        ! local
        integer :: nw, ne, ns, nn, isc, jsc, iec, jec

        isc = 1; jsc = 1; iec = npx; jec = npy
        call get_neighbor_tile_num(n,nw,ne,ns,nn)

        if (n_src.eq.nw) then
            is = isc-ng; ie = isc-1; js = jsc; je = jec
        endif
        if (n_src.eq.ne) then
            is = iec+1; ie = iec+ng; js = jsc; je = jec
        endif
        if (n_src.eq.ns) then
            is = isc; ie = iec; js = jsc-ng; je = jsc-1
        endif
        if (n_src.eq.nn) then
            is = isc; ie = iec; js = jec+1; je = jec+ng
        endif

    end subroutine get_neighbor_bounds
    !===========================================================================
    subroutine get_neighbor_tile_num(n, nw, ne, ns, nn)
        ! call get_neighbor_tile_num((in)n,(out)nw,(out)ne,(out)ns,(out)nn)
        ! get the neighbor tile number of the cubed sphere tile of number n
        ! nw => west tile; ne => east tile;
        ! ns => south tile; nn => north tile
        integer, intent(in) :: n
        integer, intent(out) :: nw,ne,ns,nn

        if ( mod(n,2).eq.0 ) then
          nn = mod(n+0,6)+1
          ne = mod(n+1,6)+1
          ns = mod(n+3,6)+1
          nw = mod(n+4,6)+1
        else
          ne = mod(n+0,6)+1
          nn = mod(n+1,6)+1
          nw = mod(n+3,6)+1
          ns = mod(n+4,6)+1
        endif

    end subroutine get_neighbor_tile_num
    !===========================================================================
end module global_grid_util_mod

