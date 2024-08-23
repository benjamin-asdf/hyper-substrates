(ns animismic.a
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.data.json :as json]
            [quil.middleware :as m]
            [ftlm.vehicles.art.lib :refer [*dt*] :as lib]
            [ftlm.vehicles.art.defs :as defs]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [fastmath.random :as fm.rand]
            [quil.core :as q]))

(def max-grid-size 30)
(def alphabet (into [] (range 5)))

(def letter->color
  [:orange :cyan :heliotrope :green-yellow :magenta])

(defn env [_state] {})

(defn draw-state
  [state]
  (def state state)


  (apply q/background (lib/->hsb (-> state :controls :background-color)))

  (q/stroke-weight 1)
  (q/stroke 0.3)
  (lib/draw-entities state)
  )

(defn update-entity
  [entity state env]
  (-> entity
      (lib/update-body state)
      lib/brownian-motion
      lib/friction
      lib/dart-distants-to-middle
      lib/move-dragged
      lib/update-rotation
      lib/update-position
      (lib/update-sensors env)
      lib/activation-decay
      lib/activation-shine
      lib/shine
      lib/update-lifetime))

(defn update-state-inner
  [state]
  ;; state
  (let [current-tick (q/millis)
        ;; state (update state
        ;;               :controls
        ;;               merge
        ;;               (user-controls/controls))
        dt (* (:time-speed (lib/controls))
              (/ (- current-tick (:last-tick state 0))
                 1000.0))
        state (binding [*dt* dt]
                (-> state
                    (assoc :last-tick current-tick)
                    lib/apply-update-events
                    lib/update-update-functions
                    lib/update-state-update-functions
                    lib/apply-events
                    ;; (lib/apply-events (:event-q
                    ;; state))
                    (lib/update-ents
                      #(update-entity % state (env state)))
                    lib/update-late-update-map
                    lib/transduce-signals
                    lib/track-components
                    lib/track-conn-lines
                    lib/ray-source-collision-burst
                    lib/kill-entities))]
    state))

(defn update-state
  [_]
  (let [state @lib/the-state
        state (update-state-inner state)]
    (reset! lib/the-state state)
    state))

(defn setup
  [controls]
  (q/rect-mode :center)
  (q/color-mode :hsb)
  (apply q/background (lib/->hsb (-> state :controls :background-color)))
  (let [state {:controls controls :on-update []}
        state (-> state lib/setup-version)]
    (reset! lib/the-state state)))

(defn sketch
  [{:as controls
    :keys [width height]
    :or {height 800 width 1000}}]
  (q/sketch :size [width height]
            :setup (partial setup controls)
            :update update-state
            :draw draw-state
            :features [:keep-on-top]
            :middleware [m/fun-mode]
            :mouse-pressed (comp #(reset! lib/the-state %)
                                 lib/mouse-pressed)
            :mouse-released (comp #(reset! lib/the-state %)
                                  lib/mouse-released)
            :mouse-wheel (comp #(reset! lib/the-state %)
                               lib/mouse-wheel)
            :frame-rate 30))



(defn draw-grid
  [{:as e :keys [grid-width spacing elements]}]
  (let [draw-element (fn [elm]
                       (q/with-fill (lib/->hsb
                                      ((letter->color
                                         (long elm))
                                        defs/color-map))
                                    (q/rect 0 0 5 5 5)))]
    (let [[x y] (lib/position e)]
      (doall (for [i (range (count elements))
                   :let [coll (mod i grid-width)
                         row (quot i grid-width)]]
               (let [x (+ x (* coll spacing))
                     y (+ y (* row spacing))]
                 (q/with-translation [x y]
                                     (draw-element
                                      (elements i)))))))))

(defmethod lib/setup-version :quantum-grid
  [state]
  (println (long (* (/ (q/screen-width) 5)
                    (/ (q/screen-height) 5))))
  (->
    state
    (lib/append-ents [;; (lib/->entity
                      ;;  :circle
                      ;;  {:color (:red defs/color-map)
                      ;;   :transform (lib/->transform
                      ;;   [5 5] 20 20 1)})
                     ])
    (lib/append-ents
      [(->
         (lib/->entity
           :q-grid
           {:draw-functions {:grid draw-grid}
            :elements (dtt/->tensor (repeatedly
                                      (* max-grid-size
                                         max-grid-size)
                                      #(rand-nth alphabet))
                                    :datatype
                                    :int32)
            :grid-width max-grid-size
            :spacing 5
            :transform (lib/->transform [50 50] 0 0 1)})
         (lib/live [:flip
                    (lib/every-n-seconds
                      0.2
                      (fn [e _ _]
                        (update e
                                :elements
                                (fn [t]
                                  (dtt/rotate
                                    t
                                    [(rand-nth [-1 -2 1
                                                2])])))))])
         (lib/live [:scramble
                    (lib/every-n-seconds
                      1
                      (fn [e _ _]
                        (update e
                                :elements
                                (fn [t]
                                  (dtt/->tensor
                                    (repeatedly
                                      (* max-grid-size
                                         max-grid-size)
                                      #(rand-nth alphabet))
                                    :datatype
                                    :int32)))))]))])))


(sketch
 {:background-color (:green-yellow defs/color-map)
  :width 800
  :height 800
  :time-speed 3
  :v :quantum-grid})
