(ns animismic.b
  (:require
   [animismic.lib.particles :as p]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [clojure.data.json :as json]
   [quil.middleware :as m]
   [ftlm.vehicles.art.lib :refer [*dt*] :as lib]
   [ftlm.vehicles.art.defs :as defs]
   [ftlm.vehicles.art.extended :as elib]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [fastmath.random :as fm.rand]
   [quil.core :as q]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [bennischwerdtner.pyutils :as pyutils]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]
   [bennischwerdtner.sdm.sdm :as sdm]
   [bennischwerdtner.hd.codebook-item-memory :as codebook]
   [bennischwerdtner.hd.ui.audio :as audio]
   [bennischwerdtner.hd.data :as hdd]))

;;
;; berp:
;;
;; brownian explorer resonator particle
;;

(def max-grid-size 30)

;;
(def alphabet (into [] (range 2)))

(def letter->color
  [:cyan :green-yellow :magenta])

(defn env [_state] {})

(defn draw-state
  [state]
  (q/background (lib/->hsb (-> state :controls :background-color)))
  (q/stroke-weight 1)
  (lib/draw-entities state))

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
  (q/background (lib/->hsb (-> controls
                               :background-color)))
  (let [state {:controls controls :on-update []}
        state (-> state
                  lib/setup-version)]
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
            :frame-rate 1))

(defn draw-grid
  [{:as e :keys [grid-width spacing elements draw-element]}]
  (let [[x y] (lib/position e)]
    (doall (for [i (range (count elements))
                 :let [coll (mod i grid-width)
                       row (quot i grid-width)]]
             (let [x (+ x (* coll spacing))
                   y (+ y (* row spacing))]
               (q/with-translation
                 [x y]
                 (draw-element e (elements i))))))))

;; (defn berp-retina
;;   [{:keys [pos spacing grid-width]}]
;;   (->
;;     (lib/->entity
;;       :q-grid
;;       {:draw-element
;;        ;; (fn [elm]
;;        ;;   (q/stroke-weight 6)
;;        ;;   (q/with-stroke
;;        ;;       (lib/with-alpha
;;        ;;         (lib/->hsb
;;        ;;          (:orange
;;        ;;           defs/color-map))
;;        ;;         (float elm))
;;        ;;       (q/with-fill nil
;;        ;;         (q/ellipse
;;        ;;          (rand-nth [-5 -2 0 2 5])
;;        ;;          (rand-nth [-5 -2 0 2 5])
;;        ;;          10 10))))
;;        (fn [_ elm]
;;          ;; (q/stroke-weight 6)
;;          (when-not (zero? elm)
;;            (q/with-stroke
;;                nil
;;                (q/with-fill
;;                    (lib/->hsb (:orange defs/color-map))
;;                  ;; (lib/with-alpha (lib/->hsb (:orange
;;                  ;;                             defs/color-map))
;;                  ;;   (float elm))
;;                    (q/ellipse (rand-nth [-5 -2 0 2 5])
;;                               (rand-nth [-5 -2 0 2 5])
;;                               8
;;                               8)))))
;;        :draw-functions {:grid draw-grid}
;;        :elements
;;        (dtt/->tensor
;;         (repeatedly
;;          (* max-grid-size
;;             max-grid-size)
;;          #(if (< (rand) 0.05) 1.0 0.0))
;;         :datatype
;;         :int32)
;;        :grid-width grid-width
;;        :spacing spacing
;;        :transform (lib/->transform pos 0 0 1)})))

(def grid-width 30)

(defn berp-retina
  [{:keys [pos spacing grid-width]}]
  (->
    (lib/->entity
      :q-grid
      {:draw-element
         (fn [_ elm]
           (when-not (zero? elm)
             (q/with-stroke
               nil
               (q/with-fill
                 (lib/->hsb (:orange defs/color-map))
                 (q/ellipse (rand-nth [-5 -2 0 2 5])
                            (rand-nth [-5 -2 0 2 5])
                            8
                            8)))))
       :draw-functions {:grid draw-grid}
       :elements (dtt/->tensor
                   (repeatedly
                     (* max-grid-size max-grid-size)
                     #(if (< (rand) 0.05) 1.0 0.0))
                   :datatype
                   :float32)
       :grid-width grid-width
       :particle-field
       (assoc (p/grid-field grid-width [p/brownian-update])
           :activations
             (pyutils/ensure-torch
               (dtt/->tensor
                 (repeatedly (* max-grid-size max-grid-size)
                             #(if (< (rand) 0.05) 1.0 0.0))
                 :datatype
                 :float32)))
       :spacing spacing
       :transform (lib/->transform pos 0 0 1)})
    (lib/live [:particle
               (fn [e _ _]
                 (let [e (update e
                                 :particle-field
                                 p/update-grid-field)]
                   (assoc e
                     :elements (pyutils/ensure-jvm
                                 (-> e
                                     :particle-field
                                     :activations)))))])))


(defn world-grid
  []
  (-> (lib/->entity
       :q-grid
       {:alpha 1
        :draw-element (fn [{:keys [alpha]} elm]
                        (q/with-fill (lib/with-alpha
                                       (lib/->hsb
                                        ((letter->color
                                          (long elm))
                                         defs/color-map))
                                       alpha)
                          (q/rect 0 0 15 15 5)))
        :draw-functions {:grid draw-grid}
        :elements (dtt/->tensor (repeatedly
                                 (* max-grid-size
                                    max-grid-size)
                                 #(rand-nth alphabet))
                                :datatype
                                :int32)
        :grid-width max-grid-size
        :spacing 20
        :transform (lib/->transform [50 50] 0 0 1)})
      (lib/live [:fades
                 (fn [e s _]
                   (let [speed 0.4]
                     (update e
                             :alpha
                             (fn [alpha]
                               (mod (+ alpha
                                       (* lib/*dt* speed))
                                    1)))))])))


(defmethod lib/setup-version :berp-retina
  [state]
  (-> state
      (lib/append-ents [ ;; (lib/->entity
                        ;;  :circle
                        ;;  {:color (:red
                        ;;  defs/color-map)
                        ;;   :transform
                        ;;   (lib/->transform
                        ;;   [5 5] 20 20 1)})
                        ])
      (lib/append-ents
       [
        ;; (world-grid)
        (berp-retina {:grid-width max-grid-size
                      :pos [50 50]
                      :spacing 20})])))



(sketch
 {:background-color [0 0 0]
  :width 800
  :height 800
  :time-speed 3
  :v :berp-retina})
