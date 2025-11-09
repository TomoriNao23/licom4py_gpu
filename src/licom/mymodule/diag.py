

def add_diag_methods(cls):
    """
    Add diagnostic methods to the class.
    """

    def get_global_stats(self):
        """
        Get global statistics by reducing h0max, h0min, h0mean from all MPI processes
        """
        import jax.numpy as jnp
        from mpi4py import MPI

        # Calculate local statistics on each process
        local_h0max, local_ubmax, local_vbmax = \
            float(jnp.max(self.h0[3:-3,3:-3])), \
            float(jnp.max(self.ub[3:-3,3:-3])), \
            float(jnp.max(self.vb[3:-3,3:-3]))
        local_h0min, local_ubmin, local_vbmin = \
            float(jnp.min(self.h0[3:-3,3:-3])), \
            float(jnp.min(self.ub[3:-3,3:-3])), \
            float(jnp.min(self.vb[3:-3,3:-3]))
        local_h0mean, local_ubmean, local_vbmean = \
            float(jnp.mean(self.h0[3:-3,3:-3])), \
            float(jnp.mean(self.ub[3:-3,3:-3])), \
            float(jnp.mean(self.vb[3:-3,3:-3]))
        
        # Get MPI communicator
        comm = MPI.COMM_WORLD
        
        # Perform MPI reduction operations on three scalar values
        # Global maximum: maximum of all process maximums
        global_h0max = comm.allreduce(local_h0max, op=MPI.MAX)
        global_ubmax = comm.allreduce(local_ubmax, op=MPI.MAX)
        global_vbmax = comm.allreduce(local_vbmax, op=MPI.MAX)
        
        # Global minimum: minimum of all process minimums
        global_h0min = comm.allreduce(local_h0min, op=MPI.MIN)
        global_ubmin = comm.allreduce(local_ubmin, op=MPI.MIN)
        global_vbmin = comm.allreduce(local_vbmin, op=MPI.MIN)

        # Global average: average of all process averages
        global_h0mean = comm.allreduce(local_h0mean, op=MPI.SUM) / comm.Get_size()
        global_ubmean = comm.allreduce(local_ubmean, op=MPI.SUM) / comm.Get_size()
        global_vbmean = comm.allreduce(local_vbmean, op=MPI.SUM) / comm.Get_size()

        return global_h0max, global_h0min, global_h0mean, \
            global_ubmax, global_ubmin, global_ubmean, \
            global_vbmax, global_vbmin, global_vbmean
            
    
    def print_global_diag(self):
        """
        Print global diagnostic information
        """
        from duogrid.duogrid import Duogrid as Dg

        global_h0max, global_h0min, global_h0mean, \
            global_ubmax, global_ubmin, global_ubmean, \
            global_vbmax, global_vbmin, global_vbmean = self.get_global_stats()
   
        if Dg.mp.pe == 0:
            print("                  max                  min                 mean")
            print(f"Global h0: {global_h0max}  {global_h0min}  {global_h0mean}")
            print(f"Global ub: {global_ubmax}  {global_ubmin}  {global_ubmean}")
            print(f"Global vb: {global_vbmax}  {global_vbmin}  {global_vbmean}")
            print("------------------------------------------------")

        
        return None
        
    cls.get_global_stats = get_global_stats
    cls.print_global_diag = print_global_diag
    return cls