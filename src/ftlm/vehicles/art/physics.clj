(ns ftlm.vehicles.art.physics
  (:require [bennischwerdtner.pyutils :as pyutils :refer
             [*torch-device*]]
            [fastmath.core :as fm]
            [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py. py..] :as py]
            [ftlm.vehicles.art.lib :as lib :refer [*dt*]]))

(do
  (require-python '[torch :as torch])
  (require-python '[builtins :as builtins]))

;; (def G 6.67e-11)
(def G 6.67)
(def softening 1e-5)
(def max-force 2e2)

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
     (torch/div
      (torch/einsum "i,j->ji" masses masses)
      (torch/pow distances 2)
      ;; (torch/add
      ;;  ;; (torch/pow softening 2)
      ;;  )
      )
      (py..
          (mul_ mask)
          (mul_ G)
          (clamp_max_ max-force)))))

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

(defn friction
  [velocities dt]
  (py.. velocities (mul_ (fm/pow 0.95 dt))))





(defn physics-update-2d
  [state]
  (def state state)
  (let [ents (filter :mass (lib/entities state))
        dt *dt*
        positions (torch/tensor
                   (into [] (map lib/position) ents)
                   :device
                   *torch-device*)
        masses (torch/tensor (into [] (map :mass) ents)
                             :device
                             *torch-device*)
        velocities
        (torch/tensor
         (into [] (map #(:velocity2d % [0 0])) ents)
         :device
         *torch-device*)
        gravity-forces (gravity positions masses)
        velocities (update-velocities velocities
                                      gravity-forces
                                      masses
                                      dt)
        velocities (friction velocities dt)
        positions
        (update-positions positions velocities dt)
        ents (map (fn [e p v]
                    (-> e
                        (assoc :velocity2d v)
                        (assoc-in [:transform :pos] p)))
                  ents
                  (pyutils/torch->jvm positions)
                  (pyutils/torch->jvm velocities))]
    ;; (def ents ents)
    (-> state
        (update :eid->entity
                merge
                (into {} (map (juxt :id identity)) ents)))))







;; (require '[libpython-clj2.python.protocols])

;; (libpython-clj2.python.protocols/python-type (torch/tensor (vec [[0 0] [0 0] [0 0]])))

;; (defmethod libpython-clj2.python.protocols/pyobject->jvm
;;   :tensor [_ obj]
;;   (pyutils/torch->jvm obj))

;; (py/->jvm (torch/tensor (vec [[0 0] [0 0] [0 0]])))
