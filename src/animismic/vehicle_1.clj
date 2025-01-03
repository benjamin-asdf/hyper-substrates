(ns
    animismic.vehicle-1
    (:require
     [animismic.lib.particles :as p]
     [fastmath.core :as fm]
     [clojure.java.io :as io]
     [clojure.string :as str]
     [clojure.data.json :as json]
     [quil.middleware :as m]
     [ftlm.vehicles.art.lib :refer [*dt*] :as lib]
     [ftlm.vehicles.art.defs :as defsc]
     [ftlm.vehicles.art.extended :as elib]
     [tech.v3.datatype.unary-pred :as unary-pred]
     [tech.v3.datatype.functional :as f]
     [tech.v3.datatype :as dtype]
     [tech.v3.tensor :as dtt]
     [quil.core :as q]
     [ftlm.vehicles.art.defs :as defs]
     [tech.v3.datatype.functional :as f]
     [tech.v3.datatype :as dtype]
     [tech.v3.tensor :as dtt]
     [tech.v3.datatype.bitmap :as bitmap]
     [fastmath.random :as fm.rand]
     ;; [bennischwerdtner.hd.binary-sparse-segmented :as hd]
     [bennischwerdtner.hd.core :as hd]
     [bennischwerdtner.pyutils :as pyutils]
     [tech.v3.datatype.unary-pred :as unary-pred]
     [tech.v3.datatype.argops :as dtype-argops]
     [bennischwerdtner.sdm.sdm :as sdm]
     [bennischwerdtner.hd.item-memory :as item-memory]
     [bennischwerdtner.hd.impl.item-memory-torch :as item-memory-torch]
     [bennischwerdtner.hd.codebook-item-memory :as codebook]
     [bennischwerdtner.hd.ui.audio :as audio]
     [bennischwerdtner.hd.data-next :as hdd]
     [animismic.lib.blerp :as b]
     [animismic.lib.particles-core :as pe]
     [ftlm.vehicles.cart :as cart]
     [animismic.lib.vehicles :as v]
     [libpython-clj2.require :refer [require-python]]
     [libpython-clj2.python :refer [py. py..] :as py]))

(defn draw-state
  [state]
  (q/background (lib/->hsb (-> state
                               :controls
                               :background-color)))
  (q/stroke-weight 0)
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
  [state dt current-tick]
  (let [env (lib/env state)
        state (binding [*dt* dt]
                (->
                  state
                  (update :t (fnil inc 0))
                  (assoc :last-tick current-tick)
                  ;; -----------------
                  lib/kill-entities
                  ;; -------------------
                  lib/apply-update-events
                  lib/update-update-functions
                  lib/update-state-update-functions
                  lib/apply-events
                  (lib/update-ents-parallel
                    #(update-entity % state env))
                  lib/update-late-update-map
                  lib/transduce-signals
                  ;; those 2 are heavy,
                  lib/track-components
                  lib/track-conn-lines
                  ;; also heavy:
                  lib/update-collisions))]
    state))

(defn update-state
  [_]
  (let [current-tick (q/millis)
        ;; state (update state
        ;;               :controls
        ;;               merge
        ;;               (user-controls/controls))
        dt (*
            (:time-speed (lib/controls))
            (/ (- current-tick
                  (:last-tick @lib/the-state 0))
               1000.0))]
    (lib/update-timers dt)
    (swap! lib/the-state update-state-inner dt current-tick)))

(defn setup
  [controls]
  (q/frame-rate 60)
  (q/rect-mode :center)
  (q/color-mode :hsb)
  ;; (q/background (lib/->hsb (-> controls
  ;;                              :background-color)))
  (let [state {:controls controls :on-update []}
        state (-> state lib/setup-version)]
    (reset! lib/the-state state)))

(defn sketch
  [{:as controls
    :keys [width height]
    :or {height 800 width 1000}}]
  (q/sketch
   :size
   ;; hard coding my monitor, else it was going to another monitor
   [2560 1920]
   ;; [1000 1000]
   ;; [500 500]
   :setup (partial setup controls)
   :update #'update-state
   :draw #'draw-state
   ;; :features [:keep-on-top]
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
    (assoc e
           :intensity (+ 10 (* 20 (:intensity-factor e))))))


(defn ->ray-source
  ([] (->ray-source (lib/rand-on-canvas-gauss 0.2)))
  ([pos]
   (let [[e] (lib/->ray-source
               {:color (:mint defs/color-map)
                :intensity 30
                :intensity-factor 1
                :kinetic-energy 1
                :on-collide-map
                {:burst (lib/cooldown 5 lib/burst)}
                :particle? true
                :on-drag-start-map
                {:survive
                 (fn [e s k]
                   (dissoc e :lifetime))}

                :on-double-click-map
                {:orient-towards-me
                 (fn [e s k]
                   {:updated-state
                    (lib/update-ents s
                                     (fn [ent]
                                       (lib/orient-towards
                                        ent
                                        (lib/position e))))}
                   )
                 }

                :pos pos
                :scale 0.75
                :shinyness nil})]
     (->
       e
       #_(lib/live
           [:rays
            (lib/every-n-seconds
              1.5
              (fn [e s _]
                {:updated-state
                   (let [bodies (into []
                                      (filter :body?
                                        (lib/entities s)))
                         a (first (shuffle bodies))
                         b e]
                     (when (and a b)
                       (->
                         s
                         (lib/append-ents
                           [(merge
                              (lib/->connection-bezier-line
                                a
                                b)
                              {:lifetime (lib/normal-distr
                                           2
                                           1)})]))))}))])
       #_(lib/live [:colors
                    (lib/every-n-seconds
                      0.5
                      (fn [e s k]
                        (assoc e
                          :color (defs/color-map
                                   (rand-nth
                                     [:green-yellow
                                      :heliotrope])))))])
       #_(lib/live [:kinetic
                    (lib/every-n-seconds
                      5
                      (fn [e s k]
                        (assoc e
                          :kinetic-energy
                            (lib/normal-distr 5 2))))])
       (lib/live
         [:circular-shine-radio
          (lib/every-n-seconds
            (fn [] (lib/normal-distr (/ 1.5 3) (/ 1.5 3)))
            (fn [ray s k]
              {:updated-state
                 (lib/append-ents
                   s
                   [(let [e (lib/->circular-shine-1 ray)]
                      (->
                        e
                        (assoc :color (lib/with-alpha
                                        (:yellow
                                          defs/color-map)
                                        0))
                        (assoc :stroke-weight 3)
                        (assoc :stroke (:color ray))
                        (assoc :on-update
                                 [(lib/->grow
                                    (* 2
                                       (+ 1
                                          (:intensity-factor
                                            ray
                                            0))))])
                        (assoc :lifetime (lib/normal-distr
                                           3
                                           (Math/sqrt
                                             3)))))])}))])
       #_(lib/live
           [:circular-shine-field
            (lib/every-n-seconds
              (fn [] (lib/normal-distr (/ 1.5 3) (/ 1.5 3)))
              (fn [ray s k]
                {:updated-state
                   (lib/append-ents
                     s
                     [(let [e (lib/->circular-shine-1 ray)]
                        (-> e
                            (assoc :color
                                     ((rand-nth
                                        [:black :hit-pink
                                         :cyan :heliotrope])
                                       defs/color-map))
                            ;; (assoc :stroke-weight
                            ;; 3)
                            ;; (assoc :stroke (:color
                            ;; ray))
                            (assoc
                              :on-update
                                [(lib/->grow
                                   (* 2
                                      (+ 1
                                         (:intensity-factor
                                           ray
                                           0))))])
                            (assoc :lifetime
                                     (lib/normal-distr
                                       10
                                       (Math/sqrt
                                         5)))))])}))])
       (lib/live [:intensity-osc update-intensity-osc])))))

(defn add-ray-source
  [state]
  (lib/append-ents state [(->ray-source)]))


(defn vehicle-1
  []
  (let [cart
          (cart/->cart
            {:body {:color (:navajo-white defs/color-map)
                    :scale 0.5
                    :stroke-weight 0
                    :vehicle-feel-color? true}
             :components
               [[:cart/motor :ma
                 {:anchor :bottom-right
                  :corner-r 5
                  :hidden? true
                  :on-update [(lib/->cap-activation)]
                  :rotational-power 0.02}]
                [:cart/motor :mb
                 {:anchor :bottom-left
                  :corner-r 5
                  :hidden? true
                  :on-update [(lib/->cap-activation)]
                  :rotational-power 0.02}]
                ;; ---------------
                [:cart/sensor :sa
                 {:anchor :top-right :modality :rays}]
                [:cart/sensor :sb
                 {:anchor :top-left :modality :rays}]
                ;; ----------------
                ;; Temperature sensor
                [:cart/sensor :hot-temperature-sensor
                 {:anchor :middle-middle
                  :hot-or-cold :hot
                  :modality :temperature}]
                ;; ----------------------------
                [:brain/neuron :arousal
                 {:arousal-neuron? true
                  :on-update [(lib/->baseline-arousal 0.2)]
                  }]
                ;; ----------------------------
                [:brain/connection :_
                 {:destination [:ref :ma]
                  :f :excite
                  :hidden? true
                  :on-update-map
                  {:gain
                   (lib/every-n-seconds
                    (fn [] (lib/normal-distr 1 1))
                    (fn [e s k]
                      (assoc-in e
                                [:transduction-model :gain]
                                (rand))))}
                  :source [:ref :arousal]}]
                [:brain/connection :_
                 {:destination [:ref :mb]
                  :f :excite
                  :hidden? true

                  :on-update-map
                  {:gain
                   (lib/every-n-seconds
                    (fn [] (lib/normal-distr 1 1))
                    (fn [e s k]
                      (assoc-in e
                                [:transduction-model :gain]
                                (rand))))}

                  :source [:ref :arousal]}]
                ;; ----------------------------
                [:brain/connection :_
                 {:decussates? false
                  :destination [:ref :ma]
                  ;; :f :excite

                  :f (lib/->weighted -10)

                  :hidden? true
                  :source [:ref :sa]}]
                [:brain/connection :_
                 {:decussates? false
                  :destination [:ref :mb]

                  :f (lib/->weighted -10)
                  ;; :f :excite
                  :hidden? true
                  :source [:ref :sb]}]
                ;; ----------------------------
                [:brain/connection :_
                 {:decussates? false
                  :destination [:ref :mb]
                  :f :excite
                  :hidden? true
                  :source [:ref :hot-temperature-sensor]}]
                [:brain/connection :_
                 {:decussates? false
                  :destination [:ref :ma]
                  :f :excite
                  :hidden? true
                  :source [:ref :hot-temperature-sensor]}]
                ;; ----------------------------
                [:brain/connection :_
                 {:decussates? true
                  :destination [:ref :ma]
                  :f :excite
                  :hidden? true
                  :on-update-map
                    {:gain (fn [e s k]
                             (assoc-in e
                               [:transduction-model :gain]
                               #(* 2 %)))}
                  :source [:ref :sb]}]
                [:brain/connection :_
                 {:decussates? true
                  :destination [:ref :mb]
                  :f :excite
                  :hidden? true
                  :on-update-map
                    {:gain (fn [e s k]
                             (assoc-in e
                               [:transduction-model :gain]
                               #(* 2 %)))}
                  :source [:ref :sa]}]
                ;; ----------------------------
               ]})]
    cart))

(defn ->vehicle-field-1
  [grid-width]
  (merge
   (p/grid-field grid-width
                 [p/attenuation-update
                  p/vacuum-babble-update p/decay-update
                  p/brownian-update p/reset-weights-update
                  p/reset-excitability-update])
   {:activations (pyutils/ensure-torch
                  (dtt/->tensor
                   (repeatedly (* grid-width grid-width)
                               #(fm.rand/flip 0.5))
                   :datatype
                   :float32))
    :attenuation-factor 0
    :decay-factor (/ 1 10)
    :size grid-width
    :vacuum-babble-factor (/ 1 20)}))

;; (filter key (group-by :particle-field-id (lib/entities @lib/the-state)))

(defn activation-flash
  [e base-color high-color kont]
  (let [sin (elib/sine-wave-machine 10 2000)
        end-timer (lib/set-timer 1)]
    (lib/live e
              [:flash
               (fn [e s k]
                 (if (lib/rang? end-timer)
                   (-> e
                       (update :on-update-map dissoc k)
                       kont)
                   ;; (update e :color lib/with-alpha
                   ;; (sin))
                   (assoc e
                     :color (q/lerp-color
                              (lib/->hsb base-color)
                              (lib/->hsb high-color)
                              (sin)))))])))

(defn ->vehicle-field
  [entities]
  (let [append-vehicles (into [] (map :id) entities)
        grid-width (long (Math/sqrt (count
                                      append-vehicles)))]
    (-> (lib/->entity :vehicle-field
                      {:particle-field (->vehicle-field-1
                                         grid-width)})
        (lib/live pe/particle-update)
        (lib/live
          [:vehicle-field
           (lib/every-n-seconds
             (fn [] (lib/normal-distr 1 1))
             (fn [e s k]
               (let [vehicle-activation
                       (pyutils/ensure-jvm
                         (p/read-activations
                           (:particle-field e)))]
                 {:updated-state
                    (reduce
                      (fn [s [id activation]]
                        #_(assoc-in s
                            [:eid->entity id :color]
                            (if (zero? activation)
                              ((rand-nth [:black])
                                defs/color-map)
                              ((rand-nth [:cyan :hit-pink])
                                defs/color-map)))
                        (if (zero? activation)
                          s
                          (->
                            s
                            (update-in
                              [:eid->entity id]
                              activation-flash
                              (:navajo-white defs/color-map)
                              (:green-yellow defs/color-map)
                              (fn [e]
                                (assoc e
                                  :color (:navajo-white
                                           defs/color-map)))
                              ;; (if (zero? activation)
                              ;;   ((rand-nth [:black])
                              ;;   defs/color-map)
                              ;;   ((rand-nth [:cyan
                              ;;   :hit-pink])
                              ;;    defs/color-map))
                            ))))
                      s
                      (map vector
                        append-vehicles
                        vehicle-activation))})))]))))

(defn +vehicle-field
  [state entities]
  (-> state
      (lib/append-ents [(->vehicle-field entities)])))

(defn setup-internal
  [state]
  (-> state
      (assoc-in
       [:on-update-map :update-colors]
       (lib/every-n-seconds
        (let [last-temp (atom 0)]
          (fn [] (lib/normal-distr 0.1 0.1)))
        (let [black? (atom false)]
          (fn [s _]
            (let [color (defs/color-map
                          (if @black?
                            :black
                            (rand-nth [:hit-pink
                                       :heliotrope
                                       :green-yellow
                                       :white :cyan])))
                  _ (swap! black? not)]
              (lib/update-ents
               s
               (fn [ent]
                 (if-not (:update-colors1? ent)
                   ent
                   (if (not @black?)
                     (assoc ent
                            :color (defs/color-map :cyan))
                     (assoc ent
                            :color defs/black))))))))))))


(defmethod lib/setup-version :vehicle-1
  [state]
  (setup-internal state))


(comment
  (swap! lib/event-queue (fnil conj []) add-ray-source)
  (swap! lib/event-queue (fnil conj []) setup-internal)
  (swap! lib/event-queue (fnil conj [])
    (fn [s]
      (update-in s [:controls :time-speed] (fnil * 1) 1.1)))
  (swap! lib/event-queue (fnil conj [])
    (fn [s]
      (update-in s [:controls :time-speed] (fnil * 1) 0.9)))
  (swap! lib/event-queue (fnil conj [])
    (fn [s]
      (update-in s [:controls :time-speed] (fnil * 1) 1.1)))
  (swap! lib/event-queue (fnil conj [])
    (fn [s]
      (update-in s [:controls :time-speed] (constantly 3))))
  (swap! lib/event-queue (fnil conj []) (kill-some 1))
  (swap! lib/event-queue (fnil conj []) (kill-some 0.2)))


(defonce vehicle-feel (atom (hd/seed)))

;; (defonce color-codebook
;;   (item-memory-torch/->codebook-matrix
;;     (vec (hdd/clj->vsa* (list
;;                           ;; disgust
;;                           :green-yellow
;;                           ;; anger
;;                           :heliotrope
;;                             ;; neutral
;;                             :navajo-white
;;                           ;; 'fear' (explore)
;;                           :amethyst-smoke
;;                             ;; joy
;;                           :hit-pink)))))

;; (item-memory-torch/codebook-cleanup-verbose
;;  hd/default-opts
;;  color-codebook
;;  (hdd/clj->vsa* :hit-pink) 0.5)

(defonce feel->color-book
  (let [d (item-memory/codebook-item-memory-1
            (torch/stack (vec (hdd/clj->vsa*
                                (list
                                  ;; disgust
                                  :green-yellow
                                  ;; anger
                                  :heliotrope
                                    ;; neutral
                                    :navajo-white
                                  ;; 'fear'
                                  ;; (explore)
                                  :amethyst-smoke
                                    ;; joy
                                    :hit-pink)))))]
    (doseq [e (list :green-yellow
                    :heliotrope :navajo-white
                    :amethyst-smoke :hit-pink)]
      (item-memory/m-clj->vsa d e))
    d))

(comment
  (item-memory/m-cleanup* feel->color-book (hdd/clj->vsa* :hit-pink) 0.5))

(defn vehicle-feel->color [hv])

(defn vehicles
  [state]
  (let [entities (mapcat identity
                   (repeatedly 12 vehicle-1))]
    (-> state
        (lib/append-ents entities)
        ;; (+vehicle-field entities)
        )))

(defn temperature-bubble
  ([] (temperature-bubble (lib/rand-on-canvas-gauss 0.2)))
  ([pos]
   (let [e (let [d 25
                 hot-or-cold :hot
                 temp 1]
             (->
               (assoc
                (lib/->entity :circle) :transform
                (lib/->transform pos d d 5) :base-scale
                5 :no-stroke?
                true :color
                (:hit-pink defs/color-map)
                :temperature-bubble?
                true :kinetic-energy
                (lib/normal-distr 0.5 0.1) :hot-or-cold
                hot-or-cold
                :lifetime
                (lib/normal-distr 2 (Math/sqrt 2))
                :collides?
                true :on-collide-map
                {:flip-color
                 (lib/cooldown
                  (fn [] (lib/normal-distr 0.2 0.1))
                  (let [counter (atom 1)]
                    (fn [e other state k]
                      (if-not (:body? other)
                        e
                        (do (swap! counter inc)
                            (-> e
                                ;; (assoc-in
                                ;; [:transform
                                ;; :scale] (* 1.08
                                ;; @counter))
                                (assoc-in [:transform
                                           :base-scale]
                                          (* 1.08
                                             @counter))
                                (assoc
                                 :color
                                 (defs/color-map
                                   ([ ;; :deep-pink
                                     :black :cyan
                                     :hit-pink]
                                    (mod @counter
                                         2))))))))))}
                :d
                d :temp
                temp :z-index
                -10 :particle?
                true :draggable?
                false)
               (lib/live
                 [:scale-it
                  (let [sin (elib/sine-wave-machine 10
                                                    2000)]
                    (fn [e s k]
                      (let [v (sin)
                            base-scale (:base-scale e)]
                        (if (<= 0.9 v)
                          (assoc e :kill? true)
                          (-> e
                              (assoc-in [:transform :scale]
                                        (+ base-scale
                                           (sin)))))))
                    ;; (lib/live e
                    ;;           [:flash
                    ;;            (fn [e s k]
                    ;;              (if (lib/rang?
                    ;;              end-timer)
                    ;;                (-> e
                    ;;                    (update
                    ;;                    :on-update-map
                    ;;                    dissoc k)
                    ;;                    kont)
                    ;;                ;; (update e
                    ;;                :color
                    ;;                lib/with-alpha
                    ;;                ;; (sin))
                    ;;                (assoc e
                    ;;                       :color
                    ;;                       (q/lerp-color
                    ;;                               (lib/->hsb
                    ;;                               base-color)
                    ;;                               (lib/->hsb
                    ;;                               high-color)
                    ;;                               (sin)))))])
                  )])))]
     e)))

(defn temperature-bubble-spawner
  []
  (->
    (assoc
      (lib/->entity :circle) :spawner?
      true :transform
      (lib/->transform (lib/rand-on-canvas-gauss 0.4) 5 5 1)
        :no-stroke?
      true :color
      (:white defs/color-map) :kinetic-energy
      1
        ;; 0.2
        :particle?
      true :on-drag-start-map
      {:spawn-ray-source (fn [e s k]
                           {:updated-state
                              (lib/append-ents
                                s
                                [(assoc (->ray-source
                                          (lib/position e))
                                   :lifetime 3)])})}
        :draggable?
      true)
    (lib/live [:spawn
               (lib/every-n-seconds
                 (fn [] (lib/normal-distr 0.5 0.1))
                 (fn [e s k]
                   {:updated-state
                      (lib/append-ents
                        s
                        [(temperature-bubble
                           (lib/v+ (lib/position e)
                                   [(lib/normal-distr 0 50)
                                    (lib/normal-distr
                                      0
                                      50)]))])}))])
    (lib/live
      [:connections-on-drag
       (lib/cooldown
         1
         (fn [e s k]
           (if-not (:dragged? e)
             e
             {:updated-state
                (lib/append-ents
                  s
                  (for [other (into []
                                    (filter
                                      lib/validate-entity)
                                    (filter :spawner?
                                      (lib/entities s)))]
                    (let [a e
                          b other]
                      (merge (lib/->connection-line a b)
                             {:color (lib/->hsb
                                       (:hit-pink
                                         defs/color-map))
                              :lifetime (lib/normal-distr
                                          2
                                          1)}))))})))])))


;; ----------------------------------------------------
(do
  (sketch {:background-color 0
           :height nil
           :time-speed 3.5
           :v :vehicle-1
           :width nil})

  ;;
  ;;
  (swap! lib/event-queue (fnil conj []) add-ray-source)
  ;;

  (swap! lib/event-queue (fnil conj []) vehicles)
  ;; (swap! lib/event-queue (fnil conj []) vehicles)
  ;; (swap! lib/event-queue (fnil conj []) vehicles)
  ;; (swap!
  ;;  lib/event-queue (fnil conj [])
  ;;  (fn [s]
  ;;    (lib/append-ents
  ;;     s
  ;;     [(temperature-bubble)])))


  (swap!
   lib/event-queue (fnil conj [])
   (fn [s]
     (lib/append-ents
      s
      (repeatedly 10 temperature-bubble-spawner))))



  (swap!
   lib/event-queue (fnil conj [])
   (fn [s]
     (assoc-in s
               [:on-update-map :spawner]
               (lib/every-n-seconds
                1
                (fn [s k]
                  (lib/append-ents
                   s
                   [(assoc (temperature-bubble-spawner) :lifetime 2)])

                  )
                )
               )

     ;; (lib/append-ents
     ;;  s
     ;;  (repeatedly 10 temperature-bubble-spawner))
     ))




  #_(swap! lib/event-queue (fnil conj [])
           (fn [s]
             (def thes s)
             (assoc-in s
                       [:on-update-map :spawn-temperature-bubble]
                       (fn [s k]
                         (lib/append-ents s [(temperature-bubble)])))))




  ;; (swap! lib/event-queue (fnil conj []) red-guys)


  #_(swap! lib/event-queue (fnil conj [])
           (fn [s]
             (assoc-in s
                       [:on-update-map :update-colors]
                       (lib/every-n-seconds
                        (let [last-temp (atom 0)]
                          (fn [] (lib/normal-distr 0.1 0.1)))
                        (fn [s _]
                          (let [color (defs/color-map
                                        (if (zero? (fm.rand/flip 0.5))
                                          :black
                                          (rand-nth [:cyan :white])))]
                            (lib/update-ents
                             s
                             (fn [ent]
                               (if-not (:body? ent)
                                 ent
                                 (assoc ent :color color)))))))))))
;; ----------------------------------------------------

(comment
  (reset! lib/the-state {})
  (reset! lib/event-queue [])
  (filter :spawner? (lib/entities @lib/the-state)))
