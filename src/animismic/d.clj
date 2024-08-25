(ns animismic.d
  (:require
   [animismic.lib.particles :as p]
   [fastmath.core :as fm]
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

;;
;;
;; ----------------------------
;; berp layer
;;
;;
;; particle fields
;;
;;
;; ----------------------------
;; conceptronic layer
;;
;;
;;






;; __________________


(def grid-width 30)

;;
(def alphabet (into [] (range 2)))

(def letter->color
  [0
   (defs/color-map :cyan)
   (defs/color-map :green-yellow)
   (defs/color-map :magenta)])










;; --------------------

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
  ;; (when (< 10 (:t state 0))
  ;;   (q/exit))
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
                    (update :t (fnil inc 0))
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
  (q/frame-rate 20)
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
  (q/sketch
   :size [width height]
   :setup (partial setup controls)
   :update update-state
   :draw draw-state
   :features [:keep-on-top]
   :middleware [m/fun-mode]
   :title "hyper-substrates"
   :key-released
   (fn [state event]
     state)
   :mouse-pressed
   (comp
    #(reset! lib/the-state %)
    lib/mouse-pressed)

   :mouse-released (comp #(reset! lib/the-state %)
                         lib/mouse-released)
   :mouse-wheel (comp #(reset! lib/the-state %)
                      lib/mouse-wheel)
   :on-close (reset! lib/the-state nil)
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

(defn field-map
  [state]
  (into {}
        (comp (filter :particle-id)
              (map (juxt :particle-id :particle-field)))
        (lib/entities state)))

(defn berp-retina
  [{:as opts
    :keys [pos spacing grid-width color particle-id]}]
  (->
    (lib/->entity
      :q-grid
      (merge
        opts
        {:draw-element (fn [_ elm]
                         (when-not (zero? elm)
                           (q/with-stroke
                               nil
                               (q/with-fill
                                   (lib/->hsb color)
                                   (q/ellipse
                                    (rand-nth [-5 -2 0 2 5])
                                    (rand-nth [-5 -2 0 2 5])
                                    8
                                    8)))))
         :draw-functions {:grid draw-grid}
         :elements []
         :grid-width grid-width
         :particle-field
         (assoc
          (p/grid-field
           grid-width
           [
            p/vacuum-babble-update
            p/decay-update
            p/brownian-update
            p/normalize-update])
          :vacuum-babble-factor (/ 1 100)
          :decay-factor 0.2
          :size grid-width
          :activations
          (pyutils/ensure-torch
           (dtt/->tensor
            (repeatedly
             (* grid-width grid-width)
             #(if (< (rand) 0.05) 1.0 0.0))
            :datatype
            :float32)))

         :particle-id particle-id
         :spacing spacing
         :transform (lib/->transform pos 0 0 1)}))
    (lib/live
     [:particle
      (fn [e s _]
        (let [
              ;; e
              ;; (update e
              ;;         :particle-field
              ;;         p/interaction-update
              ;;         (field-map s)
              ;;         (:interactions e))
              e (update e :particle-field p/update-grid-field)
              ;; _ (q/exit)
              ]
          (assoc e
                 :elements (pyutils/ensure-jvm
                            (-> e
                                :particle-field
                                :activations)))))])
    ;; (lib/live
    ;;   [:decay-pump
    ;;    (lib/every-n-seconds
    ;;      2.5
    ;;      (fn [e s _]
    ;;        (if
    ;;          (->
    ;;            (zero? (-> e
    ;;                       :particle-field
    ;;                       :decay-factor)))
    ;;          (-> e
    ;;              (assoc-in [:particle-field
    ;;              :decay-factor]
    ;;                        0.08)
    ;;              (assoc-in [:particle-field
    ;;                         :vacuum-babble-factor]
    ;;                        0))
    ;;          (-> e
    ;;              (assoc-in [:particle-field
    ;;              :decay-factor]
    ;;                        0)
    ;;              (assoc-in [:particle-field
    ;;                         :vacuum-babble-factor]
    ;;                        (/ 1 300))))))])
    ))

(defn world-grid
  []
  (->
    (lib/->entity
      :q-grid
      {:alpha 1
       :draw-element (fn [{:keys [alpha]} elm]
                       (q/stroke-weight 0.1)
                       (q/with-stroke
                         defs/white
                         (q/with-fill
                           (lib/with-alpha (lib/->hsb
                                             (letter->color
                                               (long elm)))
                                           alpha)
                           (q/rect 0 0 15 15 0))))
       :draw-functions {:grid draw-grid}
       :elements
         (dtt/->tensor
           (dtt/reshape
             (dtt/compute-tensor
               [grid-width grid-width]
               (fn [i j]
                 (if (and (< 10 i 20) (< 10 j 20)) 1.0 0.0))
               :float32)
             [(* grid-width grid-width)]))
       :grid-width grid-width
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

(defmethod lib/setup-version :berp-retina-3
  [state]
  (-> state
      (lib/append-ents
       [
        (world-grid)
        (berp-retina
         {:color (:orange defs/color-map)
          :grid-width grid-width
          :interactions [[:attracted
                          :heliotrope]]
          :particle-id :orange
          :pos [50 50]
          :spacing 20})
        (berp-retina
         {:color (:heliotrope defs/color-map)
          :grid-width grid-width
          :interactions [[:attracted :orange]]
          :particle-id :heliotrope
          :pos [50 50]
          :spacing 20})])))

(sketch
 {:background-color 0
  :height 800
  :time-speed 3
  :v :berp-retina-3
  :width 800})


(comment
  (do (reset! lib/the-state nil) (System/gc)))
