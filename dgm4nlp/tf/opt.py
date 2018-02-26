import tensorflow as tf


def minimise(loss,
             optimizer,
             grad_clipping):
    """
    Call tf minimize potentially applying gradient clipping and other tweaks.
    
    :param loss: objective to be minimised 
    :param optimizer: a tf optimiser
    :param grad_clipping: None or a pair [floor, ceil]
    :return: optimizer state
    """

    def std_minimise():
        """Default minimisation algorithm (for a given optimiser)"""
        return optimizer.minimize(loss)

    def clipped_minimise(floor, ceil):
        """Minimise loss with gradient clipping"""
        gvs = optimizer.compute_gradients(loss)
        clipped = [(tf.clip_by_value(grad, floor, ceil), var) for grad, var in gvs]
        return optimizer.apply_gradients(clipped)

    def clipped_norm(max_gradient_norm):
        """Minimise loss with gradient clipping by global norm"""
        grads, variables =  zip(*optimizer.compute_gradients(loss))
        clipped_grads, _ = tf.clip_by_global_norm(grads, max_gradient_norm)
        return optimizer.apply_gradients(zip(clipped_grads, variables))

    if grad_clipping is None:
        optimizer_state = std_minimise()
    elif len(grad_clipping) == 1:
        optimizer_state = clipped_norm(grad_clipping[0])
    else:
        optimizer_state = clipped_minimise(grad_clipping[0], grad_clipping[1])

    return optimizer_state
