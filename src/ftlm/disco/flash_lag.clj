(ns ftlm.disco.flash-lag
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
    [quil.core :as q]
    [ftlm.vehicles.art.defs :as defs]
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
    ;; [bennischwerdtner.hd.binary-sparse-segmented :as
    ;; hd]
    [bennischwerdtner.hd.core :as hd]
    [bennischwerdtner.pyutils :as pyutils]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [bennischwerdtner.sdm.sdm :as sdm]
    [bennischwerdtner.hd.item-memory :as item-memory]
    [bennischwerdtner.hd.impl.item-memory-torch :as
     item-memory-torch]
    [bennischwerdtner.hd.codebook-item-memory :as codebook]
    [bennischwerdtner.hd.ui.audio :as audio]
    [bennischwerdtner.hd.data-next :as hdd]
    [ftlm.vehicles.art.physics :as phy]
    [animismic.lib.blerp :as b]
    [animismic.lib.particles-core :as pe]
    [ftlm.vehicles.cart :as cart]
    [animismic.lib.vehicles :as v]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py..] :as py]
    [ftlm.disco.middlewares.picture-jitter :as
     picture-jitter]
    [ftlm.vehicles.art.event-q-middleware :as evtq]
    [ftlm.disco.middlewares.time-warp :as time-warp]))

(def glyph-size 18)

(defn draw-state
  [state]
  (q/background 255)
  ;; (q/with-translation [0 0] (wave/draw))
  ;; (q/color-mode :rbg)
  ;; (q/with-translation [0 500] (wave/draw))
  ;; (if (< 1 (q/random 2)) (q/color-mode :hsb))
  (q/color-mode :hsb)
  ;; (q/fill 0 0 0 50)
  ;; (q/fill 255 255 255 50)
  ;; (if (< 1 (q/random 2))
  ;;   (q/color-mode :hsb))
  ;; (q/fill (lib/with-alpha (lib/->hsb (defs/color-map
  ;;                                      (rand-nth [;; :cyan
  ;;                                                 :black
  ;;                                                 ;; :white
  ;;                                                ]))
  ;;                                    ;; (val
  ;;                                    ;;  (rand-nth
  ;;                                    ;;   (into []
  ;;                                    ;;   defs/color-map)))
  ;;                         )
  ;;           0.5))
  ;; (q/rect 0 0 (* 2 2560) (* 2 1920))
  ;; (if (even? (mod (q/seconds) 2))
  ;;   (q/color-mode :hsb)
  ;;   (q/color-mode :rgb))
  (q/color-mode :hsb)
  (do (q/stroke-weight 0) (lib/draw-entities state))
  ;; (q/with-translation [0 -200] (wave/draw))
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

(defn update-state
  [state]
  (let [env (lib/env state)
        current-tick (q/millis)
        ;; state (update state
        ;;               :controls
        ;;               merge
        ;;               (user-controls/controls))
        dt (* (:time-speed (lib/controls))
              (/ (- current-tick (:last-tick state 0))
                 2000))]
    (binding [*dt* dt]
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
        lib/update-timers-v2))))

(defn setup
  [controls]
  (q/frame-rate 120)
  (q/rect-mode :center)
  (q/color-mode :hsb)
  (q/text-size glyph-size)
  (q/text-font (q/create-font "Fira Code Bold" glyph-size)
               glyph-size)
  (let [state {:controls controls :on-update []}
        state (-> state
                  lib/setup-version)]
    state))

(defn sketch
  [{:as controls
    :keys [width height sketch-id]
    :or {height 800 width 1000}}]
  (q/sketch
    :events-q {:sketch-id sketch-id}
    :size
      ;; hard coding my monitor, else it was going to
      ;; another monitor
      [1920 1080]
    ;; [500 500]
    :setup (comp (:setup controls identity)
                 (partial setup controls))
    :update #'update-state
    :draw #'draw-state
    ;; :features [:keep-on-top]
    :middleware [m/fun-mode m/navigation-2d
                 picture-jitter/picture-jitter
                 evtq/events-middleware time-warp/time-warp]
    :navigation-2d {:modifiers {:mouse-dragged #{:shift}
                                :mouse-wheel #{:shift}}}
    :title "hyper-substrates"
    :key-released (fn [state event] state)
    :mouse-pressed (fn [s e]
                     (if (and (q/key-pressed?)
                              (= (q/key-modifiers)
                                 #{:shift}))
                       s
                       (lib/mouse-pressed s e)))
    :mouse-dragged (fn [s e] s)
    :mouse-released (fn [s e]
                      (if (and (q/key-pressed?)
                               (= (q/key-modifiers)
                                  #{:shift}))
                        s
                        (lib/mouse-released s e)))
    :mouse-wheel (fn [s e]
                   (if (and (q/key-pressed?)
                            (= (q/key-modifiers) #{:shift}))
                     s
                     (lib/mouse-wheel s e)))))


(defn activation-flash
  [e base-color high-color kont]
  (let [sin (elib/sine-wave-machine 10 2000)
        end-timer (lib/set-timer 1)
        base-color (:navajo-white defs/color-map)]
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
    (assoc e :intensity (+ 10 (* 30 (:intensity-factor e))))))

(defn holy-shine-anim
  [e s k]
  {:updated-state
     (let [clockwise? (zero? (fm.rand/flip 0.5))]
       (->
         s
         (lib/append-ents
           (mapcat identity
             (for [p
                   (repeatedly
                    5
                    (fn []
                      (let [pos (mapv +
                                      (lib/position e)
                                      (mapv #(* 50 %)
                                            (q/random-2d)))
                            rotation ((if clockwise? + -)
                                      (lib/angle-between
                                       (lib/position e)
                                       pos)
                                      q/HALF-PI)]
                        (lib/->entity
                         :circle
                         {:color (lib/->hsb-vec
                                  defs/white)
                          :lifetime 2
                          :acceleration 100
                          ;; :angular-velocity 20
                          :transform
                          (lib/->transform
                           pos
                           20
                           20
                           1
                           rotation)
                          }))))]
               [p
                (assoc (lib/->connection-line p e)
                  :z-index -5
                  :stroke-weight (abs
                                   (lib/normal-distr 3 1)))
                (lib/->entity
                  :circle
                  {:color (lib/with-alpha (lib/->hsb
                                            defs/white)
                                          0)
                   :lifetime 0.8
                   :on-update-map
                     {:grow (fn [e s k]
                              (update-in
                                e
                                [:transform :scale]
                                +
                                (lib/normal-distr 1 1)))}
                   :stroke (lib/with-alpha
                             (lib/->hsb-vec defs/white)
                             (abs (lib/normal-distr 0 1)))
                   :stroke-weight
                     (abs (lib/normal-distr 20 10))
                   :transform
                     (let [size (abs (lib/normal-distr 5
                                                       5))]
                       (lib/->transform (lib/position e)
                                        size
                                        size
                                        1))})])))))})


(defn ->ray-source
  ([] (->ray-source (lib/mid-point)))
  ([pos]
   (let [[e]
         (lib/->ray-source
          {:activation-shine-colors
           {:high (:cyan defs/color-map)
            :low defs/black}
           :color (:white defs/color-map)
           :intensity 20
           :intensity-factor 1
           :kinetic-energy 0.2
           :mass 1e5
           :on-collide-map
           {:burst (lib/cooldown
                    (fn []
                      (lib/normal-distr 2 (q/sqrt 2)))
                    lib/burst)
            :intensity
            (fn [e other s k]
              #_(audio/play!
                 (audio/->audio
                  {:duration 0.2
                   :frequency
                   (+ 150 (* 50 (rand-int 3)))}))
              (-> e
                  (update :intensity inc)
                  (update-in [:transform :scale]
                             +
                             0.01)))}
           :on-double-click-map
           {:orient-towards-me
            (fn [e s k]
              {:updated-state (lib/update-ents
                               s
                               (fn [ent]
                                 (lib/orient-towards
                                  ent
                                  (lib/position
                                   e))))})}
           :on-drag-start-map
           {:survive (fn [e s k] (dissoc e :lifetime))}
           :particle? true
           :pos pos
           :scale 0.75
           :shinyness nil
           :transform (lib/->transform
                       (lib/rand-on-canvas-gauss 0.2)
                       100
                       100
                       1)})]
     (->
      e
      ;; (lib/live (lib/every-now-and-then 0.2
      ;; holy-shine-anim))
      (lib/live (lib/every-now-and-then
                 10
                 (fn [e s k]
                   (-> e
                       (assoc-in [:transform :scale] 1)
                       (assoc-in [:intensity] 20)
                       (assoc-in [:intensity-factor]
                                 1)))))
      (lib/live [:shine
                 (fn [e s k]
                   (lib/flash-shine-1
                    e
                    (/ (:intensity e) 200)
                    {:high (defs/color-map :white
                             ;; :cyan
                             )
                     :low defs/black}))])
      #_(lib/live
         [:rays
          (lib/every-n-seconds
           (fn [] (lib/normal-distr 1 1))
           (fn [e s _]
             {:updated-state
              (let [bodies (into []
                                 (filter :block?
                                         (lib/entities s)))]
                (let [bodies (take 5 (shuffle bodies))]
                  (reduce
                   (fn [s b]
                     (->
                      s
                      (lib/append-ents
                       [(merge
                         (lib/->connection-bezier-line
                          e
                          b)
                         {:lifetime
                          (lib/normal-distr
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
                 (assoc :lifetime
                        (* 5
                           (lib/normal-distr
                            3
                            (Math/sqrt
                             3))))))])}))])
      (lib/live [:intensity-osc update-intensity-osc])))))

;; (let [ents-s (atom nil)]
;;   (def add-ray-source
;;     (fn [state]
;;       (let [state
;;               (if @ents-s
;;                 (let [current-ents @ents-s]
;;                   (lib/update-ents
;;                     state
;;                     (fn [e]
;;                       (if-let [old-e (current-ents (:id e))]
;;                         (merge (->ray-source)
;;                                (select-keys
;;                                  ((lib/entities-by-id state)
;;                                    old-e)
;;                                  [:transform]))
;;                         e))))
;;                 (let [ents [(->ray-source)]]
;;                   (reset! ents-s (into #{} (map :id) ents))
;;                   (lib/append-ents state ents)))]
;;         state))))

(defn add-ray-source
  [state]
  (let [ents [(->ray-source)]]
    (lib/append-ents state ents)))

(defn ->confettini
  []
  {:kind (rand-nth [:circle :rect :triangle])
   :color (defs/color-map (rand-nth [:hit-pink :deep-pink
                                     :green-yellow :white
                                     :cyan]))
   :block? true
   ;; :mass 100
   :moment-of-inertia 1000
   ;; :collides? true
   :on-collide-map {:die (fn [e other s k]
                           (assoc e :lifetime 1))}
   :particle? true
   :kinetic-energy 0.2
   :transform (lib/->transform (lib/rand-on-canvas-gauss
                                 0.5)
                               20
                               20
                               (* 5 (q/random-gaussian)))})

(defn confettini [] (lib/->entity :_ (->confettini)))

(defmethod lib/setup-version :vehicle-1 [state] state)

(defonce vehicle-feel (atom (hd/seed)))

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


(defn vehicle-feel->color [hv])
(defn vehicles
  [state]
  (let [entities
        (mapcat identity
                (repeatedly 36 vehicle-1))
        ]
    (-> state
        (lib/append-ents entities)
        ;; (+vehicle-field entities)
        )))

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
            ;; ((rand-nth [:deep-pink :cyan])
            ;; defs/color-map)
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
       (lib/live [:foo
                  (lib/every-n-seconds
                    5
                    (fn [e s k] (swap! counter + 1) e))])
       (lib/live [:f
                  (lib/every-n-seconds 1.5
                                       (fn [e s k]
                                         (assoc e
                                           :kinetic-energy
                                             (rand-nth
                                               [0 1 5]))))])
       (lib/live [:scale-it
                  (let [sin (elib/sine-wave-machine 2 1000)]
                    (fn [e s k]
                      (let [v (sin)]
                        (-> e
                            (assoc-in
                              [:transform :scale]
                              (+ (* 0.5
                                    (* @counter @counter))
                                 (* 0.6 (sin))))))))])))))

(defn temperature-bubble-spawner
  []
  (->
    (assoc
      (lib/->entity :circle) :spawner?
      true :transform
      (lib/->transform (lib/rand-on-canvas-gauss 0.2) 5 5 1)
        :no-stroke?
      true :color
      (:white defs/color-map) :kinetic-energy
      0.2 :particle?
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
    (lib/live
      [:spawn-ray-source
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
                    (merge
                      (lib/->connection-bezier-line a b)
                      {:color (lib/->hsb (:hit-pink
                                           defs/color-map))
                       :lifetime
                         (lib/normal-distr 2 1)}))))}))])
    (lib/live [:spawn
               (lib/every-n-seconds
                 (fn [] (lib/normal-distr 1 0.1))
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
                      (merge
                        (lib/->connection-bezier-line a b)
                        {:color (lib/->hsb
                                  (:hit-pink
                                    defs/color-map))
                         :lifetime (lib/normal-distr
                                     2
                                     1)}))))})))])))



;; ----------------------------------------------------

(defn ->yellow-heart
  ([] (->yellow-heart (lib/rand-on-canvas-gauss 0.2)))
  ([pos]
   (let [spawn-one (fn [e s _]
                     ;; {:updated-state
                     ;;  (lib/append-ents
                     ;;   s
                     ;;   (vehicle-1
                     ;;    {:pos (lib/position e)}))}
                     )
         [e] (lib/->ray-source
              {:color (defs/color-map :navajo-white-tint)
               :intensity 30
               :intensity-factor 1
               :kinetic-energy 1
               :no-stroke? true
               :on-collide-map
               {:burst (lib/cooldown 5 lib/burst)}
               :on-double-click-map {:spawn-one spawn-one}
               :particle? true
               :pos pos
               :ray-kind :yellow-heart
               :scale 1
               :z-index 5})]
     (->
      e
      (lib/live [:spawns
                 (lib/every-now-and-then 60 spawn-one)])
      (lib/live
       [:circular-shine-field
        (lib/every-n-seconds
         (fn [] (lib/normal-distr 0.5 0.5))
         (fn [ray s k]
           {:updated-state
            (lib/append-ents
             s
             [(let [e (lib/->circular-shine-1 ray)]
                (->
                 e
                 (assoc :stroke (defs/color-map :black))
                 (assoc :stroke-weight 5)
                 (assoc :color ((rand-nth [:white])
                                defs/color-map))
                 (assoc :on-update
                        [(lib/->grow
                          (* 2
                             (+ 1
                                (:intensity-factor
                                 ray
                                 0))))])
                 (assoc :lifetime (lib/normal-distr
                                   2
                                   1))))])}))])))))


(defn flash-lag-illusion
  []
  [(-> (lib/->entity :circle
                     {:color (:cyan defs/color-map)
                      :draggable? true
                      :hidden? true
                      :transform
                        (lib/->transform [200 300] 25 25 1)
                      :unique-id :ball-1})
       (lib/live
         (fn [e s k]
           (cond (and (not (:hidden? e))
                      (not (lib/rang-v2? (:flash-timer e))))
                   e
                 (and (:hidden? e)
                      (lib/rang-v2? (:flash-timer e)))
                   (assoc e
                     :hidden? true
                     :flash-timer nil)
                 :else (let [other (lib/get-entity-by-id
                                     s
                                     :ball-moving)]
                         (if (< (lib/distance
                                  (lib/position other)
                                  (lib/position e))
                                500)
                           (assoc e
                             :hidden? false
                             :flash-timer (lib/set-timer-v2
                                            1))))))))
   (->
     (lib/->entity :circle
                   {:color (:red defs/color-map)
                    :draggable? true
                    :no-friction? true
                    :transform
                      (lib/->transform [300 100] 25 25 1 0)
                    :unique-id :ball-moving
                    :velocity 500})
     (lib/live
       (lib/every-n-seconds
         1
         (fn [e s k]
           (update-in e [:transform :rotation] + q/PI)))))])

(sketch
 {:background-color 0
  :height nil
  :setup (fn [s]
           (lib/append-ents s (flash-lag-illusion)))
  :sketch-id :s
  :time-speed 3.5
  :v :vehicle-1
  :width nil})

(comment


  (evtq/append-event! add-ray-source)
  (evtq/append-event! (fn [s] (def s s)))



  (lib/get-entity-by-id s :ball-moving)



  (evtq/append-event!
   (fn  [state]
     (lib/append-ents state
                      [(->ray-source)])))
  (evtq/append-event!
   (fn [state] (lib/append-ents state [(temperature-bubble-spawner)])))
  (do
    (defn start-jitter [state]
      (assoc-in state [:picture-jitter :jitter?] true))
    (evtq/append-event!  start-jitter))
  (do
    (defn start-jitter [state]
      (assoc-in state [:picture-jitter :jitter?] false))
    (evtq/append-event!  start-jitter))


  (evtq/append-event!
   (fn [state]
     (lib/append-ents
      state
      (repeatedly 10 confettini))))


  (evtq/append-event!
   (fn [state]
     (lib/append-ents
      state
      [(->yellow-heart)])))

  (evtq/append-event!
   (fn [state]
     (lib/append-ents
      state
      [(->yellow-heart)])))

  (do
    (defn start-jitter [state]
      (assoc-in state [:picture-jitter :jitter?] true))
    (evtq/append-event! start-jitter))


  (do
    (defn start-jitter [state]
      (assoc-in state [:picture-jitter :jitter?] false))
    (evtq/append-event! start-jitter))

  (evtq/append-event!
   (fn [state]
     (lib/live
      state
      (lib/every-now-and-then
       1
       (fn [s k]
         (assoc-in s
                   [:picture-jitter :zoom-intensity]
                   0
                   ;; (rand-nth [0.99])
                   ))))))

  (evtq/append-event!
   (fn [state]
     (lib/live
      state
      [:zoom-intensity
       (lib/every-now-and-then
        1
        (fn [s k]
          (assoc-in s
                    [:picture-jitter :zoom-intensity]
                    0
                    ;; (rand-nth [0.99])
                    )))])))

  (evtq/append-event!
   (fn [state]
     (lib/live
      state
      [:zoom-intensity
       (lib/every-now-and-then
        1
        (fn [s k]
          (assoc-in s
                    [:picture-jitter :zoom-intensity]
                    0
                    ;; (rand-nth [0.99])
                    )))])))




  (evtq/append-event!
   (fn [state]
     (lib/live
      state
      (lib/every-now-and-then
       1
       (fn [s k]
         (assoc-in s
                   [:picture-jitter :zoom-intensity]
                   (rand-nth [0.999 1.001])))))))



  (evtq/append-event!
   (fn [state]
     (lib/live
      state
      [:connections
       (lib/every-now-and-then
        0.01
        (fn [s k]
          (lib/append-ents
           s
           (into
            []
            (let [[a b]
                  (into [] (shuffle (lib/entities s)))]
              [(assoc (lib/->connection-bezier-line a b)
                      :stroke-weight 20
                      :lifetime 1)])))))])))



  (evtq/append-event!
   (fn [state]
     (lib/live
      state
      (lib/every-now-and-then
       0.1
       (fn [s k]
         (lib/append-ents
          s
          (into
           []
           (let [[a b] (into []
                             (shuffle (lib/entities s)))]
             [(assoc (lib/->connection-bezier-line a b)
                     :stroke-weight (abs
                                     (lib/normal-distr 5 5))
                     :lifetime 1)]))))))))

  (evtq/append-event!
   (fn [state]
     (lib/live state
               [:swap-circ-rect
                (lib/every-now-and-then
                 0.2
                 (fn [s k]
                   (lib/update-ents
                    s
                    (fn [e]
                      (update e
                              :kind
                              {:circle :rect
                               :rect :circle})))))])))


  (evtq/append-event!
   (fn [state]
     (lib/live
      state
      [:swap-colors
       (lib/every-now-and-then
        1
        (fn [s k]
          (lib/update-ents
           s
           (fn [e]
             (assoc e
                    :color (lib/->hsb
                            ((rand-nth
                              ;; [:black :white :red
                              ;; :cyan :orange :mint]
                              [:black :white :cyan :orange]
                              ;; [:black
                              ;;  :black
                              ;;  :white
                              ;;  :red
                              ;;  :black]
                              )
                             defs/color-map)))))))])))



  (evtq/append-event!
   (fn [state]
     (lib/live
      state
      [:swap-colors
       (lib/every-now-and-then
        3
        (fn [s k]
          (lib/update-ents
           s
           (fn [e]
             (assoc e
                    :color
                    (lib/->hsb
                     ((rand-nth
                       ;; [:black :white :red
                       ;; :cyan :orange :mint]
                       ;; [:black :white :cyan
                       ;; :mint]
                       [:black :red :black :red :black
                        :red :black :red
                        ;; :cyan
                        ])
                      defs/color-map)))))))])))





  (evtq/append-event!
   (fn [state]
     (lib/live state
               [:swap-colors
                (lib/every-now-and-then
                 1
                 (fn [s k]
                   (lib/update-ents
                    s
                    (fn [e]
                      (assoc e
                             :color
                             (lib/->hsb
                              ((rand-nth
                                [:cyan :orange :red]
                                ;; [:black :white :red
                                ;; :cyan :orange :mint]
                                ;; [:black :white :cyan
                                ;; :mint]
                                )
                               defs/color-map)))))))])))


  (evtq/append-event!
   (fn [state]
     (lib/live state
               [:swap-colors
                (lib/every-now-and-then
                 1
                 (fn [s k]
                   (lib/update-ents
                    s
                    (fn [e]
                      (assoc e
                             :color
                             (lib/->hsb
                              ((rand-nth
                                [:black :orange :mint]
                                ;; [:black :white :red
                                ;; :cyan :orange :mint]
                                ;; [:black :white :cyan
                                ;; :mint]
                                )
                               defs/color-map)))))))])))




  (evtq/append-event!
   (fn [state]
     (lib/live state
               [:swap-colors
                (lib/every-now-and-then 0.1 (fn [s k] s))])))


  (evtq/append-event!
   (fn [state]
     (lib/live state
               [:scales
                (lib/every-now-and-then
                 0.1
                 (fn [s k]
                   (lib/update-ents s
                                    (fn [e]
                                      (update-in e
                                                 [:transform :scale]
                                                 *
                                                 (rand-nth
                                                  [1.0001 0.999]))))))])))




  (evtq/append-event!
   (fn [state]
     (lib/live
      state
      [:block-color
       (lib/every-now-and-then
        1
        (fn [s k]
          (lib/update-ents
           s
           (fn [e]
             (if-not (:block? e)
               e
               (assoc e
                      :color (lib/->hsb
                              (defs/color-map
                                (rand-nth
                                 [:white :black])))))))))])))






  (evtq/append-event!
   (fn [state]
     (lib/live state
               [:ray-size
                (lib/every-now-and-then
                 0.2
                 (fn [s k]
                   (lib/update-ents
                    s
                    (fn [e]
                      (if-not (-> e
                                  :kind
                                  :circle)
                        e
                        (update-in e
                                   [:transform :scale]
                                   (rand-nth
                                    [0.9 1.1])))))))])))



  (evtq/append-event!
   #(assoc-in % [:time-warp :warps?] true))
  (evtq/append-event!
   #(assoc-in % [:time-warp :warps?] false)))