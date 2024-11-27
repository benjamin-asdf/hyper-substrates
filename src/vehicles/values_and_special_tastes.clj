(ns animismic.values-and-special-tastes
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
    [ftlm.disco.middlewares.time-warp :as time-warp]
    [ftlm.disco.wave :as wave]))

(def glyph-size 18)

(defn draw-state
  [state]
  ;; (q/background 255)
  ;; (q/with-translation [0 0] (wave/draw))
  ;; (q/color-mode :rbg)
  ;; (q/with-translation [0 500] (wave/draw))

  ;; (if (< 1 (q/random 2)) (q/color-mode :hsb))
  (q/color-mode :hsb)
  ;; (q/fill 0 0 0 50)
  ;; (q/fill 255 255 255 50)
  ;; (if (< 1 (q/random 2))
  ;;   (q/color-mode :hsb))
  (q/fill
   (lib/with-alpha
     (lib/->hsb (defs/color-map
                  (rand-nth
                   [
                    ;; :cyan
                    :black
                    ;; :white
                    ]
                   ))
                ;; (val
                ;;  (rand-nth
                ;;   (into [] defs/color-map)))
                )
     0.1))
  (q/rect 0 0 (* 2 2560) (* 2 1920))
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
                 800))]
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
  (q/frame-rate 60)
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
              :intensity 40
              :intensity-factor 1
              :kinetic-energy 2
              :mass 1e5
              :on-collide-map
                {:burst (lib/cooldown
                          (fn []
                            (lib/normal-distr 2 (q/sqrt 2)))
                          lib/burst)}
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
                           30
                           30
                           1)})]
     (->
       e
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
                      (/ (:intensity e) 10)
                      {:high (defs/color-map :white)
                       :low defs/black}))])
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
       ;; (lib/live [:intensity-osc
       ;; update-intensity-osc])
       ))))

(defn add-ray-source
  [state]
  (let [ents [(->ray-source)]]
    (lib/append-ents state ents)))

(defmethod lib/setup-version :vehicle-4a [state] state)

(defn vehicle-1 []
  (cart/->cart (cart/vehicle-4a-wires)))

(defn vehicles
  [state]
  (let [entities (mapcat identity
                   (repeatedly 50 vehicle-1))]
    (def entities entities)
    (-> state
        (lib/append-ents entities))))


(sketch {:background-color 0
         :height nil
         :setup (comp vehicles add-ray-source
                      add-ray-source)
         :sketch-id :s
         :time-speed 2
         :v :vehicle-4a
         :width nil})

(comment



  (evtq/append-event! add-ray-source)
  (evtq/append-event! vehicles)

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
