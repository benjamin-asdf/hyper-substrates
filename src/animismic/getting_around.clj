(ns animismic.getting-around
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
   [animismic.lib.blerp :as b]
   [ftlm.vehicles.cart :as cart]))

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

(defn env [state]
  {:ray-sources (into [] (filter :ray-source?) (lib/entities state))})

(defn draw-state
  [state]
  (q/background (lib/->hsb (-> state :controls :background-color)))
  (q/stroke-weight 0)
  (lib/draw-entities state))

(defn update-entity
  [entity state env]
  (->
   entity

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
   lib/update-lifetime

   ))

(defn update-state-inner
  [state dt current-tick]
  ;; state
  ;; (when (< 10 (:t state 0))
  ;;   (q/exit))
  (let [env (env state)
        state (binding [*dt* dt]
                (-> state
                    (update :t (fnil inc 0))
                    (assoc :last-tick current-tick)
                    lib/apply-update-events
                    lib/update-update-functions
                    lib/update-state-update-functions
                    lib/apply-events
                    (lib/update-ents
                      #(update-entity % state env))
                    lib/update-late-update-map
                    lib/transduce-signals
                    lib/track-components
                    lib/track-conn-lines
                    lib/update-collisions
                    lib/kill-entities))]
    state))

(defn update-state
  [_]
  (let [current-tick (q/millis)
        ;; state (update state
        ;;               :controls
        ;;               merge
        ;;               (user-controls/controls))
        dt (* (:time-speed (lib/controls))
              (/ (- current-tick
                    (:last-tick @lib/the-state 0))
                 1000.0))]
    ;; (lib/update-timers dt)
    (swap! lib/the-state update-state-inner dt current-tick)))

(defn setup
  [controls]
  (q/frame-rate 60)
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
    :size :fullscreen
    ;; [500 500]
    :setup (partial setup controls)
    :update update-state
    :draw draw-state
    :features [:keep-on-top]
    :middleware [m/fun-mode]
    :title "hyper-substrates"
    :key-released (fn [state event] state)
    :mouse-pressed (comp #(reset! lib/the-state %)
                         lib/mouse-pressed)
    :mouse-released (comp #(reset! lib/the-state %)
                          lib/mouse-released)
    :mouse-wheel (comp #(reset! lib/the-state %)
                       lib/mouse-wheel)
    :on-close (reset! lib/the-state nil)))

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

(defn update-intensity-osc
  [e s _]
  (let [speed 1
        cycle-duration 20000
        e (update
           e
           :intensity-factor
           (fn [x]
             (let [fade-factor (-> (* (/ (q/millis)
                                         cycle-duration)
                                      q/TWO-PI)
                                   (Math/sin)
                                   (Math/abs))
                   wave-value (* fade-factor
                                 (+ x (* lib/*dt* speed)))]
               wave-value)))]
    (assoc e :intensity (+ 10 (* 30 (:intensity-factor e))))))

(defmethod lib/setup-version :getting-around-1
  [state]
  (->
   state
   (assoc-in
    [:on-update-map :finds-neighbours]
    (lib/every-n-seconds
     (/ 1 4)
     (fn [s _]
       (let [bodies
             (into [] (filter :body? (lib/entities s)))
             bodies (shuffle bodies)
             [a b] bodies]
         (when (and a b)
           (-> s
               (lib/append-ents
                [(merge
                  (lib/->connection-bezier-line a b)
                  {:lifetime
                   (lib/normal-distr 2 1)})])))))))
   (assoc-in [:on-update-map :update-colors]
             (lib/every-n-seconds
              2
              (fn [s _]
                (let [color (rand-nth
                             (vec (vals
                                   defs/color-map)))]
                  (lib/update-ents
                   s
                   (fn [ent]
                     (if-not (:body? ent)
                       ent
                       (assoc ent :color color))))))))
   (lib/append-ents
    (repeatedly
     1
     (fn []
       (let
           [[e]
            (lib/->ray-source
             {:color defs/white
              :intensity 30
              :intensity-factor 1
              :kinetic-energy 1.5
              :particle? true
              :pos (lib/rand-on-canvas-gauss 0.2)
              :scale 1
              :shinyness nil})]
           (lib/live e [:intensity-osc update-intensity-osc])))))

   (lib/append-ents
    (mapcat identity
            (repeatedly
             150
             (fn []
               (let [cart
                     (cart/->cart
                      {:body {:color (:navajo-white
                                      defs/color-map)
                              :hidden? true
                              :scale 1
                              :stroke-weight 0}
                       :components
                       [[:cart/entity :_
                         {:f (fn []
                               (lib/->entity
                                :circle
                                {:anchor-position [0 0]
                                 :color defs/white
                                 :transform
                                 (lib/->transform
                                  [0 0]
                                  5
                                  5
                                  1)}))}]
                        ;;
                        [:cart/motor :ma
                         {:anchor :bottom-right
                          :corner-r 5
                          :hidden? true
                          :on-update
                          [(lib/->cap-activation)]
                          :rotational-power 0.02}]
                        [:cart/motor :mb
                         {:anchor :bottom-left
                          :corner-r 5
                          :hidden? true
                          :on-update
                          [(lib/->cap-activation)]
                          :rotational-power 0.02}]
                        ;; ---------------
                        [:cart/sensor :sa
                         {:anchor :top-right
                          :hidden? true
                          :modality :rays}]
                        [:cart/sensor :sb
                         {:anchor :top-left
                          :hidden? true
                          :modality :rays}]
                        ;; ----------------------------
                        [:brain/neuron :arousal
                         {:on-update
                          [(lib/->baseline-arousal 1)]}]
                        ;; ----------------------------
                        [:brain/connection :_
                         {:destination [:ref :ma]
                          :f :excite
                          :hidden? true
                          :source [:ref :arousal]}]
                        [:brain/connection :_
                         {:destination [:ref :mb]
                          :f :excite
                          :hidden? true
                          :source [:ref :arousal]}]
                        ;; ----------------------------
                        [:brain/connection :_
                         {:destination [:ref :ma]
                          :f (lib/->weighted -10)
                          :hidden? true
                          :source [:ref :sa]}]
                        [:brain/connection :_
                         {:destination [:ref :mb]
                          :f (lib/->weighted -10)
                          :hidden? true
                          :source [:ref :sb]}]]})]
                 cart)))))



   (lib/append-ents
    (mapcat identity
            (repeatedly
             0
             (fn []
               (let [cart
                     (cart/->cart
                      {:body {:color (:navajo-white
                                      defs/color-map)
                              :scale 1
                              :stroke-weight 0}
                       :components
                       [ ;;
                        [:cart/motor :ma
                         {:anchor :bottom-right
                          :corner-r 5
                          :on-update
                          [(lib/->cap-activation)]
                          :rotational-power 0.02}]
                        [:cart/motor :mb
                         {:anchor :bottom-left
                          :corner-r 5
                          :on-update
                          [(lib/->cap-activation)]
                          :rotational-power 0.02}]
                        ;; ---------------
                        [:cart/sensor :sa
                         {:anchor :top-right
                          :modality :rays}]
                        [:cart/sensor :sb
                         {:anchor :top-left
                          :modality :rays}]
                        ;; ----------------------------
                        [:brain/neuron :arousal
                         {:on-update
                          [(lib/->baseline-arousal 1)]}]
                        ;; ----------------------------
                        [:brain/connection :_
                         {:destination [:ref :ma]
                          :f :excite
                          :hidden? true
                          :source [:ref :arousal]}]
                        [:brain/connection :_
                         {:destination [:ref :mb]
                          :f :excite
                          :hidden? true
                          :source [:ref :arousal]}]
                        ;; ----------------------------
                        [:brain/connection :_
                         {:destination [:ref :ma]
                          :f (lib/->weighted -10)
                          :hidden? true
                          :source [:ref :sa]}]
                        [:brain/connection :_
                         {:destination [:ref :mb]
                          :f (lib/->weighted -10)
                          :hidden? true
                          :source [:ref :sb]}]]})]
                 cart)))))
   #_(lib/append-ents
      (into
       []
       (repeatedly
        1
        (fn []
          (->
           (lib/->entity
            :circle
            {:angular-acceleration 20
             :color (:green-yellow defs/color-map)
             :draggable? true
             :kinetic-energy 0
             :on-click-map
             {:change-color-of-guys
              (fn [e s _]
                {:updated-state
                 (let
                     [color
                      (rand-nth
                       (vec
                        (vals
                         defs/color-map)))]
                     (lib/update-ents
                      s
                      (fn [ent]
                        (if-not (:body? ent)
                          ent
                          (assoc ent
                                 :color color)))))})}
             :particle? true
             :transform (lib/->transform
                         (lib/mid-point)
                         20
                         20
                         1)})
           (lib/live
            [:fly-towards-ray
             (lib/every-n-seconds
              1
              (fn [e s k]
                (if (zero? (rand-int 2))
                  (-> e
                      (assoc :kinetic-energy 5))
                  (let [ray-source
                        (first (filter :ray-source?
                                       (lib/entities
                                        s)))]
                    (-> e
                        (lib/orient-towards
                         (lib/position ray-source))
                        (update :acceleration
                                (fnil + 0)
                                20)
                        (assoc :kinetic-energy
                               0))))))]))))))))

(sketch
 {:background-color 0
  :time-speed 2
  :v :getting-around-1
  :height nil
  :width nil})


(comment
  (let [ray-source
        (first (filter :ray-source? (lib/entities @lib/the-state)))]
    (swap! lib/the-state update-in [:eid->entity (:id ray-source) :intensity] (constantly 0)))
  (let [ray-source
        (first (filter :ray-source? (lib/entities @lib/the-state)))]
    (swap! lib/the-state update-in [:eid->entity (:id ray-source) :intensity] (constantly 60)))
  (-> @lib/the-state :eid->entity))








;;
;; -
;;
