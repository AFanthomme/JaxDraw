import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import numpy as np
import wandb

def log_model_health(model, grads, step):
    """
    Logs layer-wise scalars and histograms to W&B.
    Everything goes to a "debug" prefix which W&B automatically groups into UI folders.
    """

    metrics = {}
    for i, (layer, l_grad) in enumerate(zip(model.layers, grads.layers)):
        p_dict = eqx.filter(layer, eqx.is_array)
        if p_dict is None: continue
        
        layer_name = f"{i:02d}_{type(layer).__name__}"
        
        p_leaves = jtu.tree_leaves(p_dict)
        all_p = jnp.concatenate([jnp.ravel(p) for p in p_leaves])
        
        p_norm = jnp.linalg.norm(all_p).item()
        abs_min = jnp.min(jnp.abs(all_p)).item()
        abs_max = jnp.max(jnp.abs(all_p)).item()
        
        metrics[f"debug/params_norm/{layer_name}"] = p_norm
        metrics[f"debug/params_abs_min/{layer_name}"] = abs_min
        metrics[f"debug/params_abs_max/{layer_name}"] = abs_max
        metrics[f"debug/weights/{layer_name}"] = wandb.Histogram(np.array(all_p))
        
        log_abs_p = jnp.log10(jnp.abs(all_p) + 1e-45)
        metrics[f"debug/log_abs_weights/{layer_name}"] = wandb.Histogram(np.array(log_abs_p))

        if l_grad is not None:
            g_leaves = jtu.tree_leaves(eqx.filter(l_grad, eqx.is_array))
            if g_leaves: # Quick safety check for empty leaves
                all_g = jnp.concatenate([jnp.ravel(g) for g in g_leaves if g is not None])
                log_abs_g = jnp.log10(jnp.abs(all_g) + 1e-45)
                metrics[f"debug/log_abs_grads_hists/{layer_name}"] = wandb.Histogram(np.array(log_abs_g))

                g_norm = jnp.linalg.norm(all_g).item()
                
                metrics[f"debug/grads_log_norm/{layer_name}"] = jnp.log10(g_norm + 1e-45)
                metrics[f"debug/grads_lognorm_ratio/{layer_name}"] = jnp.log10(g_norm / (p_norm + 1e-8) + 1e-45)
                
    wandb.log(metrics, step=step)