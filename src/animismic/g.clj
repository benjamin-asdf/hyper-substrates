(ns animismic.g
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
    [bennischwerdtner.hd.data :as hdd]
    [animismic.lib.blerp :as b]))

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


(defn update-blerp
  [e s _]
  ;; (hd/unbind b/berp-map (:particle-id e))
  (when-let [world (first (filter :world?
                                  (lib/entities s)))]
    (let [factor

          (get
           {:green-yellow 1

            :heliotrope 0 :orange 0}
           (:particle-id e)
           0)
          ;; (b/blerp-resonator-force (:particle-id e)
          ;;                          (:blerp-map
          ;;                          glooby))
          world-activations (:elements world)
          letter 1]
      ;; ------------------------------------------
      (update-in e
                 [:particle-field]
                 p/resonate-update
                 world-activations
                 (Math/pow (+ 1 factor) 3)
                 letter))
    ;; (when-let [glooby (first (keep :glooby
    ;;                                (lib/entities
    ;;                                s)))])
    ))


;; --------------

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

(defn blerp-retina
  [{:as opts
    :keys [pos spacing grid-width color particle-id
           particle-field-opts]}]
  (->
    (lib/->entity
      :q-grid
      (merge
        opts
        {:draw-element (fn [_ elm]
                         (when-not (zero? elm)
                           (q/with-stroke
                             nil
                             (q/with-fill (lib/->hsb color)
                                          (q/ellipse
                                            0 0
                                            ;; (rand-nth
                                            ;; [-5 -2 0 2
                                            ;; 5])
                                            ;; (rand-nth
                                            ;; [-5 -2 0 2
                                            ;; 5])
                                            8 8)))))
         :draw-functions {:grid draw-grid}
         :elements []
         :grid-width grid-width
         :particle-field
           (merge (p/grid-field
                    grid-width
                    [p/attenuation-update
                     p/vacuum-babble-update p/decay-update
                     p/brownian-update
                     p/reset-weights-update
                     ;; p/reset-excitability
                     p/reset-excitability-update])
                  {:vacuum-babble-factor (/ 1 500)
                   ;; (/ 1 500)
                   ;; (/ 1 500)
                   :decay-factor 0.05
                   ;; 0.01 1/2
                   :attenuation-factor 0
                   :size grid-width
                   :activations
                     (pyutils/ensure-torch
                       (dtt/->tensor
                         (repeatedly
                           (* grid-width grid-width)
                           #(if (< (rand) 0.05) 1.0 0.0))
                         :datatype
                         :float32))}
                  particle-field-opts)
         :particle-id particle-id
         :spacing spacing
         :transform (lib/->transform pos 0 0 1)}))
    (lib/live [:blerp-resonate #'update-blerp])
    (lib/live
      [:particle
       (fn [e s _]
         (let [;; (update e
               ;;         :particle-field
               ;;         p/interaction-update
               ;;         (field-map s)
               ;;         (:interactions e))
               e (update e
                         :particle-field
                         p/update-grid-field)
               e (if (= :green-yellow (:particle-id e))
                   (update e
                           :particle-field
                           p/interact-inhibiting
                           (:heliotrope (field-map s))
                           2)
                   e)
               ;; _ (q/exit)
              ]
           (assoc e
             :elements (pyutils/ensure-jvm
                         (-> e
                             :particle-field
                             :activations)))))])))

(defn world-grid
  []
  (->
    (lib/->entity
      :q-grid
      {:alpha 0
       :draw-element (fn [{:keys [alpha]} elm]
                       (when-not (zero? elm)
                         (q/stroke-weight 0.1)
                         (q/with-stroke
                           defs/white
                           (q/with-fill
                             (lib/with-alpha
                               (lib/->hsb (letter->color
                                           (long elm)))
                               alpha)
                             (q/rect 0 0 15 15 0)))))
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
       :name :world
       :spacing 20
       :transform (lib/->transform [50 50] 0 0 1)
       :world? true})
    (lib/live
      [:fades
       (fn [e s _]
         (let [speed 1
               cycle-duration 4000]
           (update
             e
             :alpha
             (fn [alpha]
               (let [fade-factor (-> (* (/ (q/millis)
                                           cycle-duration)
                                        q/TWO-PI)
                                     (Math/sin)
                                     (Math/abs))
                     wave-value (* fade-factor
                                   (+ alpha
                                      (* lib/*dt* speed)))]
                 wave-value)))))])))

(defn glooby-view
  []
  (->
   (lib/->entity
    :glooby
    {:components [(lib/->entity
                   :circle
                   {:anchor-position [0 0]
                    :color defs/white
                    :transform
                    (lib/->transform [0 0] 20 20 1)})]
     :draw-functions
     {:glooby (fn [e]
                ;;
                ;; --------------------------------------------------
                (q/stroke-weight 1)
                (q/with-translation
                    (lib/position e)
                    (let [pairs (for [a alphabet
                                      id [:orange
                                          :heliotrope]]
                                  [a id])]
                      (doseq [[n [a id]]
                              (map-indexed vector pairs)]
                        (q/with-translation
                            [0 (* n 25)]
                            (q/with-fill (lib/->hsb
                                          (letter->color a))
                              (q/rect 0 0 20 20))
                            (q/with-fill
                                (lib/->hsb (defs/color-map id))
                                (q/ellipse 0 0 10 10))))))
                ;; ---------------------------------------------------
                )}
     :transform (lib/->transform
                 [(+ 50 50 (* (inc 20) grid-width)) 50]
                 50
                 800
                 1)

     :glooby
     {:alphabet alphabet
      :blerp-map
      (hdd/clj->vsa*
       {:heliotrope (hd/->seed)
        :orange (hd/->seed)})}
     })
   (lib/live
    [:update
     (fn [e s _]
       (when-let [world (first (filter :world? (lib/entities s)))]
         (update
          e
          :glooby
          (fn [gl]
            (b/update-glooby
             gl
             (field-map s)
             (:elements world)))))
       e)])))

(defmethod lib/setup-version :berp-retina-g
  [state]
  (->
    state
    (lib/append-ents
      [
       (world-grid)
       ;; (glooby-view)
       (-> (blerp-retina
            {:color (:green-yellow
                     defs/color-map)
             :grid-width grid-width
             :particle-field-opts
             {:vacuum-babble-factor (/ 1 500)
              :decay-factor 0.01
              ;; :production-factor 1.03
              :attenuation-factor 0}
             :particle-id :green-yellow
             :pos [50 50]
             :spacing 20})
           (lib/live
            [:produced
             (fn [e s _]
               (update e
                       :particle-field
                       p/production
                       (:particle-field e)
                       (fn [n] (+ 10 n))))]))
       (-> (blerp-retina
            {:color (:heliotrope defs/color-map)
             :grid-width grid-width
             :particle-field-opts {:attenuation-factor 1.5
                                   :decay-factor 0.1}
             :particle-id :heliotrope
             :pos [50 50]
             :spacing 20})
           (lib/live
            [:produced
             (fn [e s _]
               (update e
                       :particle-field
                       p/production
                       (:green-yellow (field-map s))
                       0.2))]

            ))
       ;; (blerp-retina {:color (:green-yellow
       ;; defs/color-map)
       ;;                :grid-width grid-width
       ;;                :interactions [[:attracted
       ;;                :orange]]
       ;;                :particle-id :green-yellow
       ;;                :pos [50 50]
       ;;                :spacing 20})
       ])))

(sketch
 {:background-color 0
  :time-speed 3
  :v :berp-retina-g
  :height 1200
  :width 1200})



;;
;; -
;;
