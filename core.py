import numpy as np

def multinormal_pdf(mean, cov, x):
    """
    Return pdf of multinormal evaluated at points x.

    :param array-like mean: (n,) mean of the distribution.
    :param array-like cov: (n,n) covariance matrix.
    :param array-like x: (l, n) coordinates where the gradient is evaluated. 
    """
    x, mean = map(np.array, (x, mean))
    r = x - mean

    # Compute determinant of covariance (use slogdet, more robust)
    det = np.exp(np.linalg.slogdet(cov)[1])

    # Compute cov^-1 x r, but use solve (more robust too!)
    alpha = np.linalg.solve(cov, r.T)
    beta = np.exp(-0.5 * np.sum(r.T * alpha, axis=0))

    return 1 / np.sqrt((2*np.pi)**len(mean) * det) * beta

def gradient_normal(mean, cov, x):
    """
    Compute the gradient of a (multi)normal distribution in m coordinates.

    :param array-like mean: (n,) mean of the distribution.
    :param array-like cov: (n,n) covariance matrix.
    :param array-like x: (l, n) coordinates where the gradient is evaluated. 
    """
    x, mean = map(np.array, (x, mean))
    r = x - mean

    # Solve C.x = b, where b is the coordinate with respect to the mean 
    alpha = np.linalg.solve(cov, r.T)
    return -alpha * multinormal_pdf(mean, cov, x)

def gradient_lognormal(mean, cov, x):
    """
    Compute the gradient of the logarithm of the  (multi)normal distribution
    in m coordinates.

    :param array-like mean: (n,) mean of the distribution.
    :param array-like cov: (n,n) covariance matrix.
    :param array-like x: (l, n) coordinates where the gradient is evaluated. 
    """
    x, mean = map(np.array, (x, mean))
    r = x - mean
    return -np.linalg.solve(cov, r.T) 

def gradient_mixturenormal(mean1, cov1, mean2, cov2, alpha, x):
    """
    Compute the gradient of a mixture of two multinormals.

    :param float alpha: a number between 0 and 1, given the weigth to the
    first normal.
    """
    a = alpha
    b = 1 - alpha
    up = a * gradient_normal(mean1, cov1, x) + \
         b * gradient_normal(mean2, cov2, x)
    return up
    

def gradient_logmixturenormal(mean1, cov1, mean2, cov2, alpha, x):
    """
    Compute the gradient of the logarithm of a mixture of two multinormals.

    :param float alpha: a number between 0 and 1, given the weigth to the
    first normal.
    """
    a = alpha
    b = 1 - alpha
    up = a * gradient_normal(mean1, cov1, x) + \
         b * gradient_normal(mean2, cov2, x)

    down = a * multinormal_pdf(mean1, cov1, x) + \
           b * multinormal_pdf(mean2, cov2, x)
    return up/down


if __name__ == '__main__':
    mean = [0, 0]
    cov = np.array([[1, 0.],[0., 1]])
    mean2 = [4, 0]
    cov2 = np.array([[2, 0.8],[0.8, 2]])

    # 1000 random points between [-10, 10]^2
    x = np.random.rand(1000, 2) * 20 - 10

    import pylab as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    # Compute gradient and norm
    #grad = gradient_lognormal(mean, cov, x)
    #grad = gradient_normal(mean, cov, x)
    grad = gradient_mixturenormal(mean, cov, mean2, cov2, 0.5, x)
    gnorm = np.sqrt(np.sum(grad**2, axis=0))
    
    ax.quiver(x[:,0], x[:,1], grad[0], grad[1], gnorm, units='xy')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    
