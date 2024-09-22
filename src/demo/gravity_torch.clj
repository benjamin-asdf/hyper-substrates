(ns demo.gravity-torch
  (:require
   [bennischwerdtner.pyutils :as pyutils :refer [*torch-device*]]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]))

(do
  (require-python '[torch :as torch])
  (require-python '[builtins :as builtins]))

(def G 6.67e-11)
(def softening 1e-5)
(def max-force 1e9)

(defn pairwise-distance
  "Computes pairwise distances between all positions"
  [positions]
  (-> (torch/subtract (torch/unsqueeze positions 0)
                      (torch/unsqueeze positions 1))
      (torch/pow 2)
      (torch/sum :dim -1)
      (torch/sqrt_)
      (py.. (add_ softening))))

(defn gravitational-force
  "Computes gravitational forces between all pairs of objects"
  [masses distances]
  (let [mask (py.. (torch/ne (torch/eye (py.. masses
                                              (size 0))
                                        :device
                                        *torch-device*)
                             1)
                   (to :dtype torch/float))]
    (->
      (torch/div (torch/einsum "i,j->ji" masses masses)
                 (torch/pow distances 2))
      (py.. (mul_ mask) (mul_ G) (clamp_max_ max-force)))))

(defn force-direction
  "Computes normalized direction vectors between all pairs of objects"
  [positions distances]
  (let [diff (torch/subtract (torch/unsqueeze positions 0)
                             (torch/unsqueeze positions 1))]
    (torch/div diff (torch/unsqueeze distances -1))))

(defn calculate-forces
  "Calculates net forces on all objects"
  [positions masses]
  (let [out (torch/zeros_like positions
                              :dtype torch/float
                              :device *torch-device*)]
    (py/with-gil-stack-rc-context
      (let [distances (pairwise-distance positions)
            forces (gravitational-force masses distances)
            directions (force-direction positions distances)
            forces
              (torch/einsum "ijk,ij->ik" directions forces)]
        (py.. out (copy_ forces))))
    out))

(defn gravity
  "Returns 2d forces."
  [positions masses]
  (calculate-forces positions masses))

(defn update-velocities
  "Updates velocities based on calculated forces"
  [velocities forces masses dt]
  (let [accelerations
          (torch/div forces (torch/unsqueeze masses 1))]
    (py.. accelerations (mul_ dt) (add_ velocities))))

(defn update-positions
  "Updates positions based on velocities"
  [positions velocities dt]
  (py.. positions (add_ (torch/mul velocities dt))))




(comment
  (let [positions (torch/tensor [[300 300] [0 0]]
                                :dtype torch/float
                                :device *torch-device*)
        distances (pairwise-distance positions)]
    distances)
  (calculate-forces (torch/tensor [[300 300] [0 0] [0 50]]
                                  :dtype torch/float
                                  :device *torch-device*)
                    (torch/tensor [100 200 100]
                                  :dtype torch/float
                                  :device *torch-device*))
  ;; tensor([[-8.6005e-12, -8.0405e-12],
  ;;         [ 5.2404e-12,  5.3884e-10],
  ;;         [ 3.3600e-12, -5.3080e-10]],
  ;;         device='cuda:0')
  )
