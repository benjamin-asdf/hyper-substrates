(ns
    animismic.vehicles-and-bubbles
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
     [ftlm.vehicles.art.physics :as phy]
     [animismic.lib.blerp :as b]
     [animismic.lib.particles-core :as pe]
     [ftlm.vehicles.cart :as cart]
     [animismic.lib.vehicles :as v]
     [libpython-clj2.require :refer [require-python]]
     [libpython-clj2.python :refer [py. py..] :as py]))

(defn draw-state
  [state]
  (q/color-mode :hsb)
  ;; (if (zero? (fm.rand/flip 0.5))
  ;;   (q/color-mode :rgb)
  ;;   (q/color-mode :hsb))
  #_(q/background
     (lib/->hsb (->
                 state
                 :controls
                 :background-color))
     ;; (lib/->hsb defs/white)
     )

  (do

    (when (zero? (fm.rand/flip 0.2))
      (q/background
       (lib/->hsb (->
                   state
                   :controls
                   :background-color))
       ;; (lib/->hsb defs/white)
       ))



    (q/stroke-weight 0)
    (lib/draw-entities state)))

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

(comment
  (/ 1000 20))

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
                 lib/update-collisions
                 ;;
                 phy/physics-update-2d
                 ))]
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


(defn activation-flash
  [e base-color high-color kont]
  (let [sin (elib/sine-wave-machine 10 2000)
        end-timer (lib/set-timer 1)
        base-color (:navajo-white defs/color-map)
        ]
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
           :intensity (+ 5 (* 10 (:intensity-factor e))))))


(defn ->ray-source
  ([] (->ray-source (lib/mid-point)))
  ([pos]
   (let [[e] (lib/->ray-source
               {:color (:white defs/color-map)
                :intensity 30
                :intensity-factor 1
                :kinetic-energy 1
                :mass 1e5
                :on-collide-map
                {:burst (lib/cooldown 5 lib/burst)}
                :on-double-click-map
                {:orient-towards-me
                 (fn [e s k]
                   {:updated-state
                    (lib/update-ents
                     s
                     (fn [ent]
                       (lib/orient-towards
                        ent
                        (lib/position e))))})}
                :on-drag-start-map
                {:survive (fn [e s k]
                            (dissoc e :lifetime))}
                :particle? true
                :pos pos
                :scale 0.75
                :shinyness nil})]
     (->
      e

      (lib/live
       [:rays
        (lib/every-n-seconds
         (fn [] (lib/normal-distr 1 1))
         (fn [e s _]
           {:updated-state
            (let [bodies (into []
                               (filter :block?
                                       (lib/entities s)))]
              (let [bodies (take 5 (shuffle
                                    bodies))]
                (reduce
                 (fn [s b]
                   (->
                    s
                    (lib/append-ents
                     [(merge
                       (lib/->connection-bezier-line
                        e
                        b)
                       {:lifetime (lib/normal-distr
                                   2
                                   1)})])))
                 s
                 bodies)))}))])




      #_(lib/live [:colors
                   (lib/every-n-seconds
                    0.5
                    (fn [e s k]
                      (assoc e
                             :color (defs/color-map
                                      (rand-nth
                                       [:green-yellow
                                        :deep-pink])))))])
      #_(lib/live [:kinetic
                   (lib/every-n-seconds
                    5
                    (fn [e s k]
                      (assoc e
                             :kinetic-energy
                             (lib/normal-distr 5 2))))])
      #_(lib/live
         [:circular-shine-radio
          (lib/every-n-seconds
           (fn [] (lib/normal-distr (/ 1.5 3) (/ 1.5 3)))
           (fn [ray s k]
             {:updated-state
              (lib/append-ents
               s
               [(let [e (lib/->circular-shine-1 ray)]
                  (-> e
                      (assoc :color
                             (lib/with-alpha
                               (:yellow
                                defs/color-map)
                               0))
                      (assoc :stroke-weight 3)
                      (assoc :stroke (:color ray))
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
                      (assoc :color ((rand-nth
                                      [:black :black
                                       ;; :hit-pink
                                       ;; :cyan
                                       ;; :deep-pink
                                       :mint])
                                     defs/color-map))
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
                              5
                              (Math/sqrt
                               2)))))])}))])


      #_(lib/live
         [:circular-shine-field
          (lib/every-n-seconds
           0.1
           ;; (fn [] (lib/normal-distr (/ 1.5 3) (/ 1.5 3)))
           (fn [ray s k]
             {:updated-state
              (lib/append-ents
               s
               [(let [e (lib/->circular-shine-1 ray)]
                  (-> e
                      (assoc :color
                             ((rand-nth
                               [
                                :black
                                ;; :hit-pink
                                ;; :cyan
                                ;; :deep-pink
                                :white])
                              defs/color-map))

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
                              5
                              (Math/sqrt
                               2)))))])}))])


      (lib/live
       [:circular-shine-field
        (lib/every-n-seconds
         (fn [] (lib/normal-distr (/ 1.5 3) (/ 1.5 3)))
         (fn [ray s k]
           {:updated-state
            (lib/append-ents
             s
             [(let [e (lib/->circular-shine-1 ray)]
                (-> e
                    (assoc
                     :color
                     ((rand-nth
                       [
                        :black
                        :hit-pink
                        :white

                        ;; :white
                        ])
                      defs/color-map))
                    (lib/live
                     (fn [e s k]
                       (->
                        e
                        (update-in [:transform :scale] * 1.1 )
                        #_(assoc-in  [:color]
                                     ((rand-nth
                                       [

                                        ;; :black
                                        :hit-pink
                                        :cyan
                                        ;; :deep-pink
                                        ;; :white
                                        ])
                                      defs/color-map)))))



                    (lib/live
                     (lib/every-n-seconds 0.1
                                          (fn [e s k]
                                            (->
                                             e
                                             (assoc-in [:color] ((rand-nth [:black :white]) defs/color-map))))))


                    (assoc :lifetime (lib/normal-distr 2 (Math/sqrt 2)))))])}))])


      (lib/live
       [:circular-shine-field-1
        (lib/every-n-seconds
         (fn [] (lib/normal-distr 1 1))
         (fn [ray s k]
           {:updated-state
            (lib/append-ents
             s
             [(let [e (lib/->circular-shine-1 ray)]
                (-> e
                    (assoc :color
                           ((rand-nth
                             [
                              :black
                              :white
                              ])
                            defs/color-map))

                    (lib/live
                     (lib/every-n-seconds
                      1
                      (fn [e s k]
                        (->
                         e
                         (update-in [:transform :scale] * 1.2 )
                         (assoc-in  [:color]
                                    (rand-nth (into []
                                                    (vals defs/color-map)))))))
                     )


                    (assoc
                     :lifetime
                     (lib/normal-distr
                      5
                      (Math/sqrt
                       2)))))])}))])
      (lib/live [:intensity-osc update-intensity-osc])))))

(defn add-ray-source
  [state]
  (lib/append-ents state [(->ray-source)]))

(defonce aggression (atom 0))

(defn vehicle-1
  []
  (let [cart
          (cart/->cart
            {:body {:color (:cyan defs/color-map)
                    :scale 0.5
                    ;; :mass 1

                    :particle? true
                    :kinetic-energy 0.1

                    :moment-of-inertia 1000
                    :stroke-weight 0
                    ;; :on-update-map

                    #_{:flash1
                       (lib/every-n-seconds
                        (fn [] (lib/normal-distr 5 1))
                        (fn [e s k]
                          (activation-flash
                           e
                           (:color e)
                           (defs/color-map (rand-nth [:cyan :deep-pink]))
                           (fn [ent]
                             (assoc ent
                                    :color
                                    (:coloor e))))))}

                    ;; :on-update-map
                    ;; {:color-aggression
                    ;;  (fn [e s k]
                    ;;    (assoc
                    ;;     e :color
                    ;;     (defs/color-map
                    ;;       (if (< 0.5 @aggression)
                    ;;         :deep-pink
                    ;;         :white
                    ;;         ))))}
                    ;; :vehicle-feel-color? true
                    }
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
                  :on-update [(lib/->baseline-arousal 1)]}]
                ;; ----------------------------
                [:brain/connection :_
                 {:destination [:ref :ma]
                  :f :excite
                  :hidden? true
                  ;; :on-update-map
                  ;; {:gain
                  ;;  (lib/every-n-seconds
                  ;;   (fn [] (lib/normal-distr 1 1))
                  ;;   (fn [e s k]
                  ;;     (assoc-in
                  ;;      e [:transduction-model :gain]
                  ;;      (+ 1 (/ (rand) 2)))))}
                  :source [:ref :arousal]}]
                [:brain/connection :_
                 {:destination [:ref :mb]
                  :f :excite
                  :hidden? true
                  ;; :on-update-map
                  ;; {:gain
                  ;;  (lib/every-n-seconds
                  ;;   (fn [] (lib/normal-distr 1 1))
                  ;;   (fn [e s k]
                  ;;     (assoc-in
                  ;;      e
                  ;;      [:transduction-model :gain]
                  ;;      (+ 1 (/ (rand) 2))
                  ;;      )))}
                  :source [:ref :arousal]}]
                ;; ----------------------------
                [:brain/connection :_
                 {:decussates? false
                  :destination [:ref :ma]
                  :f :excite
                  ;; :f (lib/->weighted -10)
                  :on-update-map
                    {:gain (lib/every-n-seconds
                             (fn [] (lib/normal-distr 1 1))
                             (fn [e s k]
                               (assoc-in e
                                 [:transduction-model :gain]
                                 -20
                                 ;; (* -20 (- 1 (/
                                 ;; @aggression 5)))
                               )))}
                  :hidden? true
                  :source [:ref :sa]}]
                [:brain/connection :_
                 {:decussates? false
                  :destination [:ref :mb]
                  ;; :f (lib/->weighted -10)
                  :f :excite
                  :on-update-map
                    {:gain (lib/every-n-seconds
                             (fn [] (lib/normal-distr 1 1))
                             (fn [e s k]
                               (assoc-in e
                                 [:transduction-model :gain]
                                 -20
                                 ;; (* -20 (- 1 (/
                                 ;; @aggression 5)))
                               )))}
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
                ;; [:brain/connection :_
                ;;  {:decussates? true
                ;;   :destination [:ref :ma]
                ;;   :f :excite
                ;;   :hidden? true
                ;;   :on-update-map
                ;;   {:gain
                ;;    (lib/every-n-seconds
                ;;     (fn [] (lib/normal-distr 1 1))
                ;;     (fn [e s k]
                ;;       (assoc-in
                ;;        e [:transduction-model :gain]
                ;;        (/ @aggression 2))))}
                ;;   :source [:ref :sb]}]
                ;; [:brain/connection :_
                ;;  {:decussates? true
                ;;   :destination [:ref :mb]
                ;;   :f :excite
                ;;   :hidden? true
                ;;   :on-update-map
                ;;   {:gain
                ;;    (lib/every-n-seconds
                ;;     (fn [] (lib/normal-distr 1 1))
                ;;     (fn [e s k]
                ;;       (assoc-in
                ;;        e [:transduction-model :gain]
                ;;        (/ @aggression 2))))}
                ;;   :source [:ref :sa]}]
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
                              ;; (:white defs/color-map)
                              (:color e)
                              (:green-yellow defs/color-map)
                              (fn [e]
                                (assoc e
                                  :color (:white
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

      (lib/append-ents
       (repeatedly
        20
        (fn []
          (lib/->entity
           :rect
           {:color (defs/color-map
                     (rand-nth [:hit-pink :deep-pink
                                :green-yellow :white
                                :cyan]))
            :block? true

            :mass 1
            :moment-of-inertia 1000
            ;; :collides? true
            :on-collide-map
            {:die
             (fn [e other s k]
               (assoc e :lifetime 1))}

            ;; :particle? true
            ;; :kinetic-energy 1
            :transform (lib/->transform
                        (lib/rand-on-canvas-gauss 0.5)
                        20
                        50
                        1)}))))
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
                                       :deep-pink
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
           (update-in s [:controls :time-speed] (fnil * 1) 0.01)))



  (swap! lib/event-queue (fnil conj [])
         (fn [s]
           (update-in s [:controls :time-speed] (fnil * 1) 1.1)))



  (swap! lib/event-queue (fnil conj [])
         (fn [s]
           (update-in s [:controls :time-speed] (constantly 5))))
  (swap! lib/event-queue (fnil conj [])
         (fn [s]
           (update-in s [:controls :time-speed] (constantly 3))))


  (swap! lib/event-queue (fnil conj []) (kill-some 1))

  (swap! lib/event-queue (fnil conj []) (lib/kill-some 0.2)))


(defonce vehicle-feel (atom (hd/seed)))

;; (defonce color-codebook
;;   (item-memory-torch/->codebook-matrix
;;     (vec (hdd/clj->vsa* (list
;;                           ;; disgust
;;                           :green-yellow
;;                           ;; anger
;;                           :deep-pink
;;                             ;; neutral
;;                             :white
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
                                  :deep-pink
                                    ;; neutral
                                    :white
                                  ;; 'fear'
                                  ;; (explore)
                                  :amethyst-smoke
                                    ;; joy
                                    :hit-pink)))))]
    (doseq [e (list :green-yellow
                    :deep-pink :white
                    :amethyst-smoke :hit-pink)]
      (item-memory/m-clj->vsa d e))
    d))

(comment
  (item-memory/m-cleanup* feel->color-book (hdd/clj->vsa* :hit-pink) 0.5))

(defn vehicle-feel->color [hv])

(defn vehicles
  [state]
  (let [entities (mapcat identity
                   (repeatedly 36 vehicle-1))]
    (-> state
        (lib/append-ents entities)
        (+vehicle-field entities))))

(defn temperature-bubble
  ([] (temperature-bubble (lib/rand-on-canvas-gauss 0.2)))
  ([pos]
   (let [d 25
         hot-or-cold :hot
         temp 1
         counter (atom 1)]
     (->
      (merge
       (lib/->entity :circle)
       {:collides? true
        :color
        ;; ((rand-nth [:deep-pink :cyan]) defs/color-map)
        ((rand-nth [:white :deep-pink]) defs/color-map)
        :d d
        :draggable? false
        :hot-or-cold hot-or-cold
        :kinetic-energy 0
        :lifetime (lib/normal-distr 10 (Math/sqrt 2))
        :no-stroke? true
        :on-collide-map
        {:flip-color
         (lib/cooldown
          0.1
          (fn [e other state k]
            (if-not (:body? other)
              e
              (do
                (swap! counter inc)
                (if (and (= (mod @counter 2) 0)
                         (zero? (fm.rand/flip 0.5)))
                  {:updated-state
                   (cond->
                       (update-in
                        state
                        [:eid->entity (:id e)]
                        (fn [e]
                          (-> e
                              (assoc :lifetime 0.1)
                              (lib/live [:grow
                                         (lib/->grow
                                          1)])
                              (update :on-update-map
                                      dissoc
                                      :scale-it))))
                       (zero? (fm.rand/flip 0.75))
                       (lib/+explosion e))}
                  (-> e
                      (assoc :lifetime
                             (lib/normal-distr
                              10
                              (Math/sqrt 2)))
                      (assoc :color
                             (defs/color-map
                               ([:deep-pink :cyan]
                                (mod @counter
                                     2))))))))))}
        :particle? true
        :temp temp
        :temperature-bubble? true
        ;; this flash from to big scale looks cool
        :transform (lib/->transform pos d d 1)
        :z-index -10})
      (lib/live [:foo (lib/every-n-seconds 5 (fn [e s k] (swap! counter + 1) e))])
      (lib/live
       [:f
        (lib/every-n-seconds
         1.5
         (fn [e s k ]
           (assoc e :kinetic-energy (rand-nth [0 1 5]))))]
       )
      (lib/live
       [:scale-it
        (let [sin (elib/sine-wave-machine 2 1000)]
          (fn [e s k]
            (let [v (sin)]
              (-> e
                  (assoc-in
                   [:transform :scale]
                   (+ (* 0.5 (* @counter @counter))
                      (* 0.6 (sin))))))))])))))

(defn temperature-bubble-spawner
  []
  (->
   (assoc
    (lib/->entity :circle) :spawner?
    true :transform
    (lib/->transform (lib/rand-on-canvas-gauss 0.4) 5 5 1)
    :no-stroke?
    true :color (:white defs/color-map)
    :kinetic-energy 0.2
    :particle?
    true
    :on-drag-start-map
    {:spawn-ray-source (fn [e s k]
                         {:updated-state
                          (lib/append-ents
                           s
                           [(assoc (->ray-source
                                    (lib/position e))
                                   :lifetime 3)])})}

    :draggable?
    true)

   (lib/live
    [
     :spawn-ray-source
     (lib/every-n-seconds
      1
      (fn [e s k]
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
              (merge (lib/->connection-bezier-line a b)
                     {:color (lib/->hsb
                              (:hit-pink
                               defs/color-map))
                      :lifetime (lib/normal-distr
                                 2
                                 1)}))))}))])

   (lib/live
    [:spawn
     (lib/every-n-seconds
      (fn [] (lib/normal-distr 1 0.1))
      (fn [e s k]
        {:updated-state
         (lib/append-ents
          s
          [(temperature-bubble
            (lib/v+ (lib/position e)
                    [(lib/normal-distr 0 50)
                     (lib/normal-distr 0 50)]))])}))])

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
                (merge (lib/->connection-bezier-line a b)
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

  ;; (swap! lib/event-queue (fnil conj []) vehicles)


  ;; (swap! lib/event-queue (fnil conj []) vehicles)


  ;; (swap! lib/event-queue (fnil conj []) vehicles)

  (swap!
   lib/event-queue (fnil conj [])
   (fn [s]
     (lib/append-ents
      s
      [(temperature-bubble)])))


  (swap!
   lib/event-queue (fnil conj [])
   (fn [s]
     (lib/append-ents
      s
      (repeatedly 1 temperature-bubble-spawner))))

  (let [v (elib/sine-wave-machine 1 10000)]
    (lib/state-on-update!
     (fn [s k]
       (reset! aggression (v))
       s
       )))






  ;; (swap!
  ;;  lib/event-queue (fnil conj [])
  ;;  (fn [s]
  ;;    (assoc-in s
  ;;              [:on-update-map :spawner]
  ;;              (lib/every-n-seconds
  ;;               5
  ;;               (fn [s k]
  ;;                 (lib/append-ents
  ;;                  s
  ;;                  [(assoc (temperature-bubble-spawner) :lifetime 5)])

  ;;                 )
  ;;               )
  ;;              )

  ;;    ;; (lib/append-ents
  ;;    ;;  s
  ;;    ;;  (repeatedly 10 temperature-bubble-spawner))
  ;;    ))




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


  (lib/state-on-update!
   (fn [s k]
     (lib/update-ents
      s
      (fn [ent]
        (if
            (:body?)
            ent
            (if (not (zero? (fm.rand/flip 0.5)))
              (assoc ent
                     :color (defs/color-map :cyan))
              (assoc ent
                     :color defs/white)))))))

  (swap! lib/event-queue (fnil conj [])
         (lib/kill-some
          0.5))


  (reset! lib/the-state {})




  (reset! lib/event-queue [])
  (filter :spawner? (lib/entities @lib/the-state))




  (swap! lib/event-queue (fnil conj [])
         (fn [s]
           (assoc-in s [:controls :time-speed] 2))))







(comment


  (defn add-blocks! []
    (swap! lib/event-queue
           (fnil conj [])
           (fn [s]
             (lib/append-ents
              s
              (repeatedly
               20
               (fn []
                 (lib/->entity
                  :rect
                  {:color (defs/color-map
                            (rand-nth [:hit-pink :deep-pink
                                       :green-yellow :white
                                       :cyan]))
                   :block? true
                   :mass 1
                   :moment-of-inertia 1000
                   ;; :collides? true
                   :on-collide-map
                   {:die
                    (fn [e other s k]
                      (assoc e :lifetime 1))}
                   ;; :particle? true
                   ;; :kinetic-energy 1
                   :transform (lib/->transform
                               (lib/rand-on-canvas-gauss 0.5)
                               20
                               50
                               1)})))))))




  (add-blocks!)









  (defn update-blocks
    [op]
    (swap! lib/event-queue (fnil conj [])
           (fn [s]
             (lib/update-ents s (fn [e]
                                  (if-not (:block? e) e (op e)))))))
  (update-blocks
   (fn [e]
     (assoc e :color defs/white)))


  (update-blocks
   (fn [e]
     (assoc-in
      e
      [:on-collide-map :foo ]
      (fn [e other s k ]
        (->
         e
         (assoc :color defs/white)
         (assoc :scale 2))))))

  (update-blocks
   (fn [e]
     (assoc
      e
      {:on-collide-map
       {:foo
        (fn [e other s k ]
          (->
           e
           (assoc :color defs/white)
           (assoc :scale 2)))}})))



  (update-blocks
   (fn [e]
     (assoc
      e
      :collides? true
      )))






  (update-blocks
   (fn [e]
     (assoc e :color (:deep-pink defs/color-map))))

  (update-blocks
   (fn [e]
     (lib/live
      e
      [:a
       (lib/every-n-seconds
        1
        (fn [e s k]
          (assoc e :color (defs/color-map
                            (rand-nth [:white :black])))))])))
  (update-blocks
   (fn [e]
     (lib/live
      e
      (let [sin (elib/sine-wave-machine 2 1000)]
        (fn [e s k]
          (let [v (sin)]
            (-> e
                (assoc-in
                 [:transform :scale]
                 (sin)))))))))




  (update-blocks
   (fn [e]
     (lib/live
      e
      (let [sin (elib/sine-wave-machine 10 500)]
        (fn [e s k]
          (let [v (sin)]
            (-> e
                (assoc-in
                 [:transform :scale]
                 (* 10 (sin))))))))))

  (def small? (atom false))















  (swap! lib/event-queue (fnil conj [])

         (fn [s]
           (swap! small? not)
           (lib/update-ents s (fn [e]
                                (update-in e  [:transform :scale] *
                                           (if @small? 0.5 2))))))

  (lib/state-on-update!
   (lib/every-n-seconds
    (fn [] (* 2 (fm.rand/frand))
      ;; (lib/normal-distr 0.2 0.1)
      )
    (fn [s k]
      (swap! small? not)
      (lib/update-ents s (fn [e]
                           (update-in e  [:transform :scale] *
                                      (if @small? 0.5 2)))))))

  )


(defn full-block
  ([])
  ([n]
   (->
    (lib/->entity
     :rect
     {:color (defs/color-map (rand-nth
                              ;; [:cyan :white]
                              [:white :heliotrope]))
      :full-block? true
      :lifetime 0.1
      :transform
      (lib/->transform [(* n 200) 0] 200 (* 2 (q/height)) 1)})
    (lib/live
     (lib/every-n-seconds 0.1
                          (fn [e s k]
                            (assoc e :color (defs/color-map (rand-nth [:cyan :white])))))))))

(comment

  (swap! lib/event-queue (fnil conj [])
         (fn [s]
           (lib/append-ents
            s
            [(full-block 1)]
            )))

  (def block-n (atom 0))
  (def dir (atom dec))

  (swap!
   lib/event-queue (fnil conj [])
   (fn [s]
     (lib/live s
               [:fo
                (lib/every-n-seconds
                 0.2
                 ;; (fn [] (lib/normal-distr 0.2 0.1))
                 (fn [s k]

                   (lib/append-ents
                    s
                    [(full-block (swap! block-n
                                        (comp #(mod % 15)
                                              @dir)))])))])))


  (swap!
   lib/event-queue (fnil conj [])
   (fn [s]
     (-> s
         (lib/live [:swap-dir


                    (lib/every-n-seconds
                     (fn [] (lib/normal-distr 5 2))
                     (fn [s k]

                       (swap! dir {dec inc inc dec})
                       s))])
         (lib/live
          [:fo
           (lib/every-n-seconds
            0.2
            ;; (fn [] (lib/normal-distr 0.2 0.1))
            (fn [s k]
              (lib/append-ents
               s
               [(full-block (swap! block-n
                                   (comp #(mod % 15)
                                         @dir)))])))]))))









  (let [block-n (atom 0)
        dir (atom dec)]
    (swap!
     lib/event-queue (fnil conj [])
     (fn [s]
       (-> s
           (lib/live [(random-uuid)
                      (lib/every-n-seconds
                       (fn [] (lib/normal-distr 5 2))
                       (fn [s k]

                         (swap! dir {dec inc inc dec})
                         s)
                       )])
           (lib/live
            [(random-uuid)
             (lib/every-n-seconds
              0.2
              ;; (fn [] (lib/normal-distr 0.2 0.1))
              (fn [s k]
                (lib/append-ents
                 s
                 [(full-block (swap! block-n
                                     (comp #(mod % 15)
                                           @dir)))])))]))))))


(comment

  (update-blocks
   (fn [e]
     (lib/live
      e
      [:a
       (lib/every-n-seconds
        1
        (fn [e s k]
          (assoc e :color (defs/color-map
                            (rand-nth [:white :black])))))])))





  (update-blocks
   (fn [e]
     (lib/live
      e
      [:lines
       (lib/every-n-seconds
        2
        (fn [e s k]
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
                (merge (lib/->connection-bezier-line a b)
                       {:color
                        (lib/->hsb
                         (:hit-pink
                          defs/color-map))
                        :lifetime (lib/normal-distr
                                   2
                                   1)}))))}


          ))])))

  )
