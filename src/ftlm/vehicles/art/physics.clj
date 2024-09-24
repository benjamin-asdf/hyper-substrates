(ns ftlm.vehicles.art.physics
  (:require [bennischwerdtner.pyutils :as pyutils :refer
             [*torch-device*]]
            [fastmath.core :as fm]
            [tech.v3.tensor :as dtt]
            [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py. py..] :as py]
            [ftlm.vehicles.art.lib :as lib :refer [*dt*]]))

;;
;; This is a toy physics engine
;; It shows a certain disregard for Newton
;;


(do
  (require-python '[torch :as torch])
  (require-python '[torch.nn.functional])
  (require-python '[torch.linalg])
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

(defn friction
  [velocities dt]
  (py.. velocities (mul_ (fm/pow 0.95 dt))))

;; ---------------------------------------------------------------

(defn directions-to-angles
  "Converts a tensor of shape (n-objects, 2) to a tensor of shape (n-objects) with the angles.
  Each angle is calculated using torch.atan2 for the y and x components."
  [directions]
  (let [x (torch/select directions 1 0)
        y (torch/select directions 1 1)]
    (torch/subtract (torch/atan2 y x) fm/HALF_PI)))

;; ---------------------------------------------------------------


;;
;; Actually, what I had before as 'browian motion' was more toy,
;; but also more cute because the vehicles moved always forward, not to the side like now
;; now the brownian motion makes you look like you are blown in the wind -
;; which makes sense Ig.
;;
;; Also interesting how much the mass influences this, massive things don't have brownian motion
;;

;; kinetic energy = 1/2 m v^2
(defn brownian-motion
  [kinetic-energies masses]
  (torch/einsum
   "ij,i->ij"
   (torch/normal 0
                 100 [(py.. kinetic-energies (size 0)) 2]
                 :device *torch-device*)
   (torch/sqrt (torch/div (torch/mul kinetic-energies 2)
                          masses))))

;; ---------------------------------------------------------------

(defn update-angular-velocity
  "Updates angular velocity based on the torque and moment of inertia"
  [angular-velocity torque moment-of-inertia dt]
  (let [angular-acceleration (torch/div torque moment-of-inertia)]
    (torch/add angular-velocity (torch/mul angular-acceleration dt))))

(defn update-angle
  "Updates the angle based on the angular velocity"
  [angle angular-velocity dt]
  (torch/add angle (torch/mul angular-velocity dt)))

(defn update-angular-accelerations
  "Updates angular accelerations based on force directions and current object rotations.
  If the object is aligned with the force, the angular acceleration is zero.
  Otherwise, the angular acceleration is proportional to the angular difference."
  [rotations angular-accelerations moment-of-inertias
   forces]
  (let [force-angles (directions-to-angles forces)
        angle-diff (torch/subtract force-angles rotations)
        torque (torch/mul
                 (torch.linalg/vector_norm forces :dim 1)
                 (torch/sin angle-diff))
        new-angular-accelerations
          (torch/add angular-accelerations
                     (torch/div torque moment-of-inertias))]
    new-angular-accelerations))


;; -----------------------------------------------

#_(defn motor-forces
    [ent]
    (let [effectors (lib/motors ent state)]
      {:motor-force (reduce + (map #(:activation % 0) effectors))
       :motor-torque
       (transduce (map lib/effector->angular-acceleration)
                  +
                  effectors)}))

;; ----------------------------------------------------



(defn physics-update-2d
  [state]
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
        kinetic-energies
        (torch/tensor
         (into [] (map #(:kinetic-energy % 0) ents))
         :device
         *torch-device*)
        brownian-forces (brownian-motion kinetic-energies
                                         masses)
        gravity-forces (gravity positions masses)
        forces (torch/sum (torch/stack [gravity-forces
                                        brownian-forces])
                          :dim
                          0)
        velocities
        (update-velocities velocities forces masses dt)
        velocities (friction velocities dt)
        positions
        (update-positions positions velocities dt)
        rotations (torch/tensor (into []
                                      (map (fn [e]
                                             (->
                                              e
                                              :transform
                                              :rotation))
                                           ents))
                                :device
                                *torch-device*)
        angular-accelerations
        (torch/tensor
         (into []
               (map #(:angular-acceleration % 0) ents))
         :device
         *torch-device*)
        moment-of-inertias
        (torch/tensor
         (into []
               (map #(:moment-of-inertia % 1000) ents))
         :device
         *torch-device*)
        ;; -------------------------------------------
        ;; Updating angular velocities a little bit
        ;; so the objects rotate towards a gravity
        ;; source
        angular-accelerations
        (update-angular-accelerations
         rotations
         angular-accelerations
         moment-of-inertias
         forces)
        ents (doall
              (map (fn [e p v a]
                     (-> e
                         (assoc :velocity2d v)
                         (assoc :angular-acceleration a)
                         (assoc-in [:transform :pos] p)))
                   ents
                   (pyutils/torch->jvm positions)
                   (pyutils/torch->jvm velocities)
                   (pyutils/torch->jvm
                    angular-accelerations)))]
    (-> state
        (update
         :eid->entity
         merge
         (into {} (map (juxt :id identity)) ents)))))
