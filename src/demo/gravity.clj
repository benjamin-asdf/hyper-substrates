(ns demo.gravity
  (:require [quil.core :as q]
            [quil.middleware :as m]))

(def G 6.67e-11)
(def scale-factor 1)

(defn distance [[x1 y1] [x2 y2]]
  (Math/sqrt (+ (Math/pow (- x2 x1) 2)
                (Math/pow (- y2 y1) 2))))

(defn gravitational-force [m1 m2 r]
  (/ (* G m1 m2) (Math/pow r 2)))

(defn force-direction [[x1 y1] [x2 y2]]
  (let [dx (- x2 x1)
        dy (- y2 y1)
        d (Math/sqrt (+ (* dx dx) (* dy dy)))]
    [(/ dx d) (/ dy d)]))

(defn apply-force [object force]
  (let [ax (/ (first force) (:mass object))
        ay (/ (second force) (:mass object))]
    (-> object
        (update-in [:velocity 0] + ax)
        (update-in [:velocity 1] + ay))))

(defn update-position [object]
  (-> object
      (update-in [:position 0] + (first (:velocity object)))
      (update-in [:position 1] + (second (:velocity object)))))

(defn setup []
  (q/frame-rate 60)
  {:objects
   [{:position [200 200] :velocity [0 0.5] :mass 1 :radius 10}
    {:position [400 200] :velocity [0 -0.5] :mass 1 :radius 10}
    {:position [300 300] :velocity [0 0] :mass 1e12 :radius 5}]})

(defn calculate-forces [objects]
  (for [obj objects]
    (let [other-objects (remove #(= % obj) objects)
          forces (for [other other-objects]
                   (let [r (/ (distance (:position obj) (:position other)) scale-factor)
                         force (gravitational-force (:mass obj) (:mass other) r)
                         [dx dy] (force-direction (:position obj) (:position other))]
                     [(* force dx) (* force dy)]))]
      (reduce (fn [obj force]
                (apply-force obj force))
              obj
              forces))))

(defn update-state [state]
  (update state :objects
          (fn [objects]
            (->> objects
                 calculate-forces
                 (map update-position)))))

(defn draw [state]
  (q/background 0)
  (q/fill 255)
  (doseq [obj (:objects state)]
    (let [[x y] (:position obj)]
      (q/ellipse x y (* 2 (:radius obj)) (* 2 (:radius obj))))))

(q/defsketch gravity-sim
  :title "Multi-object Gravity Simulation"
  :size [600 400]
  :setup setup
  :update update-state
  :draw draw
  :features [:keep-on-top]
  :middleware [m/fun-mode])
