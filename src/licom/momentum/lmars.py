import jax.numpy as jnp

from backend.calculation.field import Field
from duogrid.duogrid import Duogrid as Dg
from momentum.barotropic import jax
from operators.poly import scalar_interpolation_xy, vector_interpolation
from operators.poly import vector_interpolation_ew, vector_interpolation_ns
import functools


class LMARS:

    celerity_x : Field.datatype
    celerity_y : Field.datatype

    vel_vis_x : Field.datatype
    vel_vis_y : Field.datatype

    def __init__(self):

        self.celerity_x = Field.new('2d')
        self.celerity_y = Field.new('2d')
        self.vel_vis_x = Field.new('2d')
        self.vel_vis_y = Field.new('2d')

    def get_celerity(self, h : Field.datatype) -> None:
        """
        Get the celerity of the ocean current.
        TODO: need to add the topography.

        Formula:
            a_x = sqrt(g * h_x + dzph_x)
            a_y = sqrt(g * h_y + dzph_y)

        a = 10.0 m/swhere a < 10.0 m/s
        """
        # get the interpolated ocean depth
        hx, hy = scalar_interpolation_xy(h)

        # get the celerity of the ocean current
        self.celerity_x = self.celerity_x.at[:,:].set(
            jnp.sqrt(9.8 * hx + Dg.dzph_x)
        )
        self.celerity_y = self.celerity_y.at[:,:].set(
            jnp.sqrt(9.8 * hy + Dg.dzph_y)
        )

        # limit the celerity limit to 10.0 m/s
        self.celerity_x = self.celerity_x.at[:,:].set(
            jnp.where(
                self.celerity_x[:,:]<10.0,
                10.0,
                self.celerity_x[:,:]
            )
        )
        self.celerity_y = self.celerity_y.at[:,:].set(
            jnp.where(
                self.celerity_y[:,:]<10.0,
                10.0,
                self.celerity_y[:,:]
            )
        )

        return None


    def get_vel_vis_2d(self, h : Field.datatype) -> None:
        """
        Get the Viscosity.velocity of the ocean current.
        TODO: need to add the topography.

        Formula:
            vel_vis_x = g * (h_e(i-1) - h_w(i)) * 0.5 / a
            vel_vis_y = g * (h_n(j-1) - h_s(j)) * 0.5 / a
        """

        # get the interpolated ocean depth
        he, hw , hn, hs = vector_interpolation(h)

        # get the Viscosity.velocity of the ocean current
        self.vel_vis_x = self.vel_vis_x.at[3:-2,:].set(
            4.9 / self.celerity_x[3:-2,:] * (
                he[2:-3,:] - hw[3:-2,:]
            )
        )
        self.vel_vis_y = self.vel_vis_y.at[:,3:-2].set(
            4.9 / self.celerity_y[:,3:-2] * (
                hn[:,2:-3] - hs[:,3:-2]
            )
        )

        return None


    def get_pgf_vis_2d(self, u : Field.datatype, v : Field.datatype) -> None:
        """
        Get the Viscosity.pressure gradient force of the ocean current.

        Formula:
            vel_vis_x = 0.5 * a * (u_e(i-1) - u_w(i))
            vel_vis_y = 0.5 * a * (v_n(j-1) - v_s(j))
        """
        # get the interpolated ocean current
        ue, uw = vector_interpolation_ew(u)
        vn, vs = vector_interpolation_ns(v)
        
        # get the Viscosity.pressure of the ocean current
        self.vel_vis_x = self.vel_vis_x.at[3:-2,:].set(
            0.5 * self.celerity_x[3:-2,:] * (
                ue[2:-3,:] - uw[3:-2,:]
            )
        )
        self.vel_vis_y = self.vel_vis_y.at[:,3:-2].set(
            0.5 * self.celerity_y[:,3:-2] * (
                vn[:,2:-3] - vs[:,3:-2]
            )
        )

        # get the Viscosity.pressure gradient force of the ocean current
        self.vel_vis_x = self.vel_vis_x.at[3:-3,3:-3].set(
            Dg.rdx[3:-3,3:-3] * (
                self.vel_vis_x[4:-2,3:-3] - self.vel_vis_x[3:-3,3:-3]
            )
        )
        self.vel_vis_y = self.vel_vis_y.at[3:-3,3:-3].set(
            Dg.rdy[3:-3,3:-3] * (
                self.vel_vis_y[3:-3,4:-2] - self.vel_vis_y[3:-3,3:-3]
            )
        )

        return None


    def add_vel_vis(self, u : Field.datatype, v : Field.datatype) -> None:
        """
        Add the Viscosity.velocity of the ocean current.
        """
        u = u.at[:,:].set(u[:,:] + self.vel_vis_x[:,:])
        v = v.at[:,:].set(v[:,:] + self.vel_vis_y[:,:])

        return None

    
    def add_pgf_vis(self, x : Field.datatype, y : Field.datatype) -> None:
        """
        Add the Viscosity.pressure gradient force of the ocean current.
        """
        x = x.at[:,:].set(x[:,:] - self.vel_vis_x[:,:])
        y = y.at[:,:].set(y[:,:] - self.vel_vis_y[:,:])     

        return None

