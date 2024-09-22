(ns demo.gravity-sketch
  (:require
   [bennischwerdtner.pyutils :as pyutils :refer
    [*torch-device*]]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]
   [demo.gravity-torch :as g]
   [quil.core :as q]
   [quil.middleware :as m]))

(do
  (require-python '[torch :as torch])
  (require-python '[builtins :as builtins]))

(defn setup-objects
  [n]
  (let [->obj (fn []
                {:color [(* 256 (rand)) (* 256 (rand))
                         (* 256 (rand))]
                 :mass 1e10
                 :position [(* 600 (rand)) (* 400 (rand))]
                 :radius 10
                 :velocity [0 0]})]
    (concat [{:mass 1e13
              :position [300 300]
              :radius 5
              :velocity [0 0]}]
            (repeatedly n ->obj))))


;; Example "data orientedness" (game development)
;;

(defn setup
  []
  (q/frame-rate 60)
  (let [objects (setup-objects 1500)]
    {:masses (torch/tensor (vec (map :mass objects))
                           :device
                           *torch-device*)
     :objects objects
     :positions (torch/tensor (vec (map :position objects))
                              :device
                              *torch-device*)
     :velocities (torch/tensor (vec (map :velocity objects))
                               :device
                               *torch-device*)}))

(defn update-physics
  [{:keys [masses velocities positions]}]
  (let [forces (g/gravity positions masses)
        velocities
          (g/update-velocities velocities forces masses 1)
        positions
          (g/update-positions positions velocities 1)]
    {:masses masses
     :positions positions
     :positions-jvm (pyutils/torch->jvm positions)
     :velocities velocities}))

(defn update-state
  [state]
  (merge state (update-physics state)))

(defn draw
  [state]
  (q/background 0)
  (q/fill 255)
  (time (doall (map (fn [obj [x y]]
                      (q/ellipse x
                                 y
                                 (* 2 (:radius obj))
                                 (* 2 (:radius obj)))
                      ;; (q/with-fill [256 0 0])
                    )
                 (:objects state)
                 (:positions-jvm state)))))

(q/defsketch gravity-sim
  :title "Multi-object Gravity Simulation"
  :size [600 400]
  :setup setup
  :update update-state
  :draw draw
  :features [:keep-on-top]
  :middleware [m/fun-mode])

(comment
  (def objects (setup-objects 2))
  (let [forces (g/gravity (map :position objects)
                          (map :mass objects))
        masses (torch/tensor (vec (map :mass objects))
                             :device
                             *torch-device*)
        velocities (torch/tensor (vec (map :velocity
                                        objects))
                                 :device
                                 *torch-device*)
        velocities
          (g/update-velocities velocities forces masses 1)
        positions (torch/tensor (vec (map :position
                                       objects))
                                :device
                                *torch-device*)
        positions
          (g/update-positions positions velocities 1)]
    (for [[x y] (pyutils/torch->jvm positions)] [x y])))
