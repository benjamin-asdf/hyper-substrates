(ns animismic.grids
  (:require
   [arc.arc-grid :as grid]
   [arc.ca :as ca]
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
  (def state state)
  (q/color-mode :hsb)
  (q/fill
   (lib/with-alpha (lib/->hsb (defs/color-map
                                (rand-nth [:black])))
     0.1))
  (q/rect 0 0 (* 2 2560) (* 2 1920))
  (q/color-mode :hsb)
  (do (q/stroke-weight 0) (lib/draw-entities state)))

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
   ;; [1920 1080]
   [1000 1000]
   ;; [500 500]
   :setup (partial setup controls)
   :update #'update-state
   :draw #'draw-state
   ;; :features [:keep-on-top]
   :middleware [m/fun-mode
                ;; m/navigation-2d
                ;; picture-jitter/picture-jitter
                evtq/events-middleware
                time-warp/time-warp
                ]
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



;; https://arcprize.org/

(def arc-colors
  {:black "#000000"
   :blue "#1E93FF"
   :blue-light "#87D8F1"
   :gray "#555555"
   :gray-light "#999999"
   :green "#4FCC30"
   :magenta "#E53AA3"
   :magenta-light "#ff7bcc"
   :maroon "#921231"
   :offwhite "#C0C0C0"
   :orange "#FF851B"
   :red "#F93C31"
   :white "#EEEE"
   :yellow "#FFDC00"})

(def arc-symbols
  [ ;; 0
   {:color :black}
   ;; 1
   {:color :blue}
   ;; 2
   {:color :red}
   ;; 3
   {:color :green}
   ;; 4
   {:color :yellow}
   ;; 5
   {:color :gray-light}
   ;; 6
   {:color :magenta}
   ;; 7
   {:color :orange}
   ;; 8
   {:color :blue-light}
   ;; 9
   {:color :maroon}])

(defn draw-arc-grid
  [e]
  (let [element-size (/ (:size e)
                        (max (py.. (:elements e) (size 0))
                             (py.. (:elements e)
                                   (size 1))))]
    (q/rect-mode :corner)
    (q/with-stroke
      (when (< 0 (:border-size e 0))
        (lib/->hsb (-> arc-colors
                       :gray)))
      (q/stroke-weight (:border-size e 0))
      (q/with-translation
        [0 0]
        ;; (lib/position e)
        (doseq [[i row] (map-indexed vector (:elements e))]
          (doseq [[j elm] (map-indexed vector row)]
            (let [elm (py.. elm item)]
              (q/with-fill (lib/->hsb
                             (case elm
                               0 (lib/with-alpha (:white defs/color-map) 0)
                               -1 (:white defs/color-map)
                               (or (:color e)
                                   (arc-colors
                                     (:color (arc-symbols
                                              elm))))))
                           (q/rect (* j element-size)
                                   (* i element-size)
                                   element-size
                                   element-size)))))))))

(defn window-grid
  []
  (let [grid (torch/tensor [[5 5 5 5 0 5 5 5 5 0 5 5 5 5]
                            [5 5 5 5 0 0 5 5 0 0 5 0 0 5]
                            [5 0 0 5 0 0 5 5 0 0 5 0 0 5]
                            [5 0 0 5 0 5 5 5 5 0 5 5 5 5]])]
    (-> (lib/->entity :arc-grid
                      {:border-size 0
                       :draw-functions {:f draw-arc-grid}
                       ;; :color defs/white
                       :elements grid
                       :initial-grid grid
                       :size (q/width)
                       :transform
                       (lib/->transform [0 0] 1 1 1)
                       :z-index -5})
        (lib/live
         (lib/every-now-and-then
          0.5
          (fn [e s k]
            (update e
                    :elements
                    (fn [grid]
                      (if (zero? (py.. (torch/sum
                                        grid)
                                   item))
                        (grid/rand-grid)
                        grid)
                      ;; (:initial-grid e)
                      )))))
        (lib/live (lib/every-now-and-then
                    0.1
                    (fn [e s k]
                      (update e
                              :elements
                              (fn [grid]
                                (ca/update-function
                                 grid)))))))))



(defn one-grid
  ([] (one-grid false))
  ([isbig?]
   (let [grid (torch/tensor [[0 1 0] [0 1 0] [0 1 0]])
         ;; isbig? (zero? (fm.rand/flip 0.8))
         size (if isbig? (q/width) (rand-nth [100 200]))
         initial-pos (lib/rand-on-canvas-gauss 0.2)]
     (-> (lib/->entity
           :arc-grid
           {:border-size 1
            :draw-functions {:f draw-arc-grid}
            :elements grid
            :initial-grid grid
            :kinetic-energy 0.8
            :particle? true
            :size size
            :transform (lib/->transform initial-pos 1 1 1)})
         (lib/live (lib/every-now-and-then
                     0.2
                     (fn [e s k]
                       (update
                         e
                         :border-size
                         (fn [size]
                           (if (zero? (fm.rand/flip 0.2))
                             (max 0
                                  (+ size
                                     (rand-nth [-10 -1 1 5
                                                -5 10])))
                             0))))))
         (lib/live (lib/every-now-and-then
                     0.1
                     (fn [e s k]
                       (update e
                               :elements
                               (fn [grid]
                                 (ca/update-function
                                   grid))))))
         (lib/live (lib/every-now-and-then
                     1
                     (fn [e s k]
                       (-> e
                           (assoc :elements (:initial-grid
                                              e))
                           (assoc-in [:transform :pos]
                                     initial-pos)))))))))

(defn add-grids
  [state]
  (lib/append-ents state (repeatedly 20 one-grid)))

(do
  (sketch {:background-color 0
           :sketch-id :s
           :time-speed 3.5
           :v :grids})
  ;; (evtq/append-event! add-grids)
  (evtq/append-event!
   (fn [s] (lib/append-ents s [(window-grid)]))))

(comment




  (def example-grid
    (-> (arc.arc-grid/puzzle) :test first :input))


  (do
    (defn start-jitter [state]
      (assoc-in state [:picture-jitter :jitter?] true))
    (evtq/append-event!  start-jitter))
  (do
    (defn start-jitter [state]
      (assoc-in state [:picture-jitter :jitter?] false))
    (evtq/append-event!  start-jitter))





  (do
    (defn start-jitter [state]
      (assoc-in state [:picture-jitter :jitter?] false))
    (evtq/append-event! start-jitter))


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
