;; generated with claude, seems to get the job done
(ns demo.gravity
  (:require [quil.core :as q]
            [quil.middleware :as m]))


(def G 6.67e-11)
;; (def G 1)
(def scale-factor 1)
(def softening 1e-5)
(def forcecap 100)

(defn distance [[x1 y1] [x2 y2]]
  (Math/sqrt (+ (Math/pow (- x2 x1) 2)
                (Math/pow (- y2 y1) 2))))

(defn gravitational-force
  [m1 m2 r]
  (min (/ (* G m1 m2)
          (Math/pow r 2))
       1e9))

(defn force-direction [[x1 y1 :as x] [x2 y2 :as y]]
  (let [dx (- x2 x1)
        dy (- y2 y1)
        d (distance x y)]
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

(defn setup
  []
  (q/frame-rate 60)
  (let [->obj (fn []
                {:mass 1e10
                 :position [(* 600 (rand)) (* 400 (rand))]
                 :radius 10
                 :velocity [0 0]})]
    {:objects
     (concat [{:mass 1e13
               :position [300 300]
               :radius 5
               :velocity [0 0]}]
             (repeatedly 1 ->obj))}))

(defn calculate-forces-1
  [objects]
  (for [[i obj] (map-indexed vector objects)]
    (let [forces (for [[j other] (map-indexed vector
                                              objects)]
                   (if (= i j)
                     [0 0]
                     (let [r (/ (distance (:position obj)
                                          (:position other))
                                scale-factor)
                           force (gravitational-force
                                   (:mass obj)
                                   (:mass other)
                                   r)
                           [dx dy] (force-direction
                                     (:position obj)
                                     (:position other))]
                       [(* force dx) (* force dy)])))]
      forces)))

(defn calculate-forces
  [objects]
  (map (fn [obj forces]
         (let [f (reduce (fn [[acc-x acc-y] [x y]]
                           [(+ acc-x x) (+ acc-y y)])
                   forces)]
           (apply-force obj f)))
    objects
    (calculate-forces-1 objects)))

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

(comment
  (force-direction [0 0] [100 100])
  (force-direction [0 0] [100 100])
  (distance [0.7071067811865475 0.7071067811865475] [0 0])
  (calculate-forces)

  (map
   (fn [forces]
    )
   (calculate-forces-1 [{:mass 100 :position [300 300]}
                        {:mass 200 :position [0 0]}
                        {:mass 100 :position [0 50]}]))


  ([-8.600470486671992E-12 -8.040466559358933E-12]
   [5.240446922793635E-12 5.388404469227936E-10]
   [3.360023563878358E-12 -5.307999803634347E-10]))
