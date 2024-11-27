(ns ftlm.vehicles.cart
  (:require
   [emmy.env :as e]
   [clojure.walk :as walk]
   [bennischwerdtner.hd.ui.audio :as audio]
   [ftlm.vehicles.art.extended :as elib]
   [ftlm.vehicles.art.defs :as defs]
   [ftlm.vehicles.art.lib :as lib]
   [quil.core :as q :include-macros true]))

(defn ->rand-sensor-pair-plans
  [motor-left motor-right]
  (let [modality (rand-nth [:rays :smell :temperature])
        sensor-left-opts {:anchor :top-left
                          :modality modality
                          :shuffle-anchor? (#{:smell}
                                            modality)}
        sensor-left-opts
        (merge
         sensor-left-opts
         (when (= modality :smell)
           {:activation-shine-colors
            {:high (:misty-rose defs/color-map)
             :low (:heliotrope defs/color-map)}


            :fragrance (rand-nth [:oxygen
                                  :organic-matter])})
         (when (= modality :temperature)
           {:hot-or-cold (rand-nth [:hot :cold])}))
        sensor-right-opts (assoc sensor-left-opts
                                 :anchor :top-right)
        decussates? (rand-nth [true false])
        sensor-left-id (random-uuid)
        sensor-right-id (random-uuid)
        transduction-fn (rand-nth [:excite :inhibit])]
    (case modality
      :temperature
      [[:cart/sensor sensor-left-id
        (assoc sensor-left-opts
               :anchor :middle-middle
               :activation-shine-colors
               ({:cold {:high {:h 196 :s 26 :v 100}
                        :low defs/white}
                 :hot {:high (:hit-pink defs/color-map)
                       :low defs/white}}
                (:hot-or-cold sensor-left-opts)))]
       [:brain/connection :_
        {:bezier-line (lib/rand-bezier 5)
         :destination [:ref motor-left]
         :f transduction-fn
         :source [:ref sensor-left-id]}]
       [:brain/connection :_
        {:bezier-line (lib/rand-bezier 5)
         :destination [:ref motor-right]
         :f transduction-fn
         :source [:ref sensor-left-id]}]]
      [[:cart/sensor sensor-left-id sensor-left-opts]
       [:cart/sensor sensor-right-id sensor-right-opts]
       [:brain/connection :_
        {:bezier-line (lib/rand-bezier 5)
         :destination [:ref motor-left]
         :f transduction-fn
         :source [:ref
                  (if decussates?
                    sensor-right-id
                    sensor-left-id)]}]
       [:brain/connection :_
        {:bezier-line (lib/rand-bezier 5)
         :destination [:ref motor-right]
         :f transduction-fn
         :source [:ref
                  (if decussates?
                    sensor-left-id
                    sensor-right-id)]}]])))

(defn ->love-wires
  [motor-left motor-right sensor-opts]
  (let [sensor-left-opts (merge sensor-opts {:anchor :top-left})
        sensor-right-opts (assoc sensor-left-opts :anchor :top-right)
        sensor-left-id (random-uuid)
        sensor-right-id (random-uuid)
        decussates? false]
    [[:cart/sensor sensor-left-id sensor-left-opts]
     [:cart/sensor sensor-right-id sensor-right-opts]
     [:brain/connection :_
      {:destination [:ref motor-left]
       :f :inhibit
       :source [:ref (if decussates? sensor-right-id sensor-left-id)]}]
     [:brain/connection :_
      {:destination [:ref motor-right]
       :f :inhibit
       :source [:ref (if decussates? sensor-left-id sensor-right-id)]}]]))

(defn random-multi-sensory
  [sensor-pair-count]
  (fn [{:as opts :keys [baseline-arousal]}]
    {:body (merge opts
                  {:color-of-the-mind
                     (rand-nth [:cyan :hit-pink
                                :navajo-white :sweet-pink
                                :woodsmoke :mint
                                :midnight-purple])})
     :components
       (into [[:cart/motor :motor-left
               {:activation-shine-colors
                  {:high (:misty-rose defs/color-map)}
                :activation-shine-speed 0.5
                :anchor :bottom-left
                :corner-r 5
                :on-update [(lib/->cap-activation)]
                :rotational-power 0.02}]
              [:cart/motor :motor-right
               {:activation-shine-colors
                  {:high (:misty-rose defs/color-map)}
                :activation-shine-speed 0.5
                :anchor :bottom-right
                :corner-r 5
                :on-update [(lib/->cap-activation)]
                :rotational-power 0.02}]
              [:brain/neuron :arousal
               {:activation-shine true
                :activation-shine-colors
                  {:high (:red defs/color-map)}
                :nucleus :arousal
                :on-update [(lib/->baseline-arousal
                              (or baseline-arousal 0.8))]}]
              [:brain/connection :_
               {:destination [:ref :motor-left]
                :f rand
                :hidden? true
                :source [:ref :arousal]}]
              [:brain/connection :_
               {:destination [:ref :motor-right]
                :f rand
                :hidden? true
                :source [:ref :arousal]}]]
             ;; (->love-wires :motor-left :motor-right
             ;; {:modality :smell
             ;; :fragrance :oxygen})
             (mapcat identity
               (repeatedly sensor-pair-count
                           (fn []
                             (->rand-sensor-pair-plans
                               :motor-right
                               :motor-left)))))}))

(def body-plans
  {:multi-sensory (random-multi-sensory 6)})

(defn shuffle-anchor [{:keys [shuffle-anchor?] :as e}]
  (if-not shuffle-anchor?
    e
    (let [[x y] (lib/anchor->trans-matrix (:anchor e))
          anch-pos
          [(lib/normal-distr x 0.2)
           (lib/normal-distr y 0.12)]]
      (assoc e :anchor-position anch-pos))))

(def builders
  {:brain/connection
   (comp
    lib/->connection
    #(walk/prewalk-replace
      {:excite lib/excite
       :inhibit lib/inhibit}
      %))
   :brain/neuron
   (comp elib/with-electrode-sensitivity lib/->neuron)
   :cart/body
   (fn [opts]
     (lib/->body
      (merge
       {:color (:sweet-pink defs/color-map)
        :corner-r 10
        :darts? true
        :draggable? true
        :on-update-map
        {:indicator
         (lib/every-n-seconds
          1
          (fn [e s _]
            (if (= (:id e) (:id (:selection s)))
              (assoc e
                     :stroke-weight 4
                     :stroke (:amethyst-smoke
                              defs/color-map))
              (dissoc e :stroke-weight :stroke))))}
        :pos (lib/rand-on-canvas-gauss 0.3)
        :rot (* (rand) q/TWO-PI)
        :scale 1}
       opts)))
   :cart/motor (comp elib/with-electrode-sensitivity lib/->motor)
   :cart/sensor
   (comp
    ;; elib/with-electrode-sensitivity
    shuffle-anchor
    lib/->sensor)})

(defmulti build-entity first)

(defmethod build-entity :cart/entity [[_ {:keys [f] :as opts}]] (merge (f) opts))

(defmethod build-entity :default [[kind opts]] ((builders kind) opts))

(defn ref? [v] (and (sequential? v) (= (first v) :ref)))

;; only have maps 1 deep right now

(defn resolve-refs
  [temp-id->ent form]
  (update-vals
    form
    (fn [v]
      (cond (ref? v)
              (or (temp-id->ent (second v))
                  (throw
                    #?(:cljs (throw
                               (js/Error.
                                 (str (second v)
                                      " is not resolved")))
                       :clj (Exception.
                              (str (second v)
                                   " is not resolved")))))
            (map? v) (resolve-refs temp-id->ent v)
            :else v))))

(defn ->cart
  [{:keys [body components]}]
  (let [body (build-entity [:cart/body body])
        {:keys [comps]}
          (reduce (fn [{:keys [comps temp-id->ent]}
                       [kind temp-id opts]]
                    (let [entity (build-entity
                                   [kind
                                    (resolve-refs
                                      temp-id->ent
                                      opts)])]
                      {:comps (into comps
                                    (if (map? entity)
                                      [entity]
                                      entity))
                       :temp-id->ent (if (= temp-id :_)
                                       temp-id->ent
                                       (assoc temp-id->ent
                                         temp-id entity))}))
            {:comps [] :temp-id->ent {}}
            components)
        comps (map #(assoc % :body (:id body)) comps)]
    (into [(assoc body
                  :components (into [] (map :id) comps))]
          comps)))

(defn vehicle-death
  [s e]
  (future
    (audio/play!
     (audio/->audio {:duration 0.1
                     :frequency 600}))
    (audio/play!
     (audio/->audio
      {:duration 0.1
       :frequency 150})))
  (let [new-e (-> e
                  (assoc :lifetime 0.2)
                  (lib/live (lib/->grow 0.1)))]
    (-> (lib/+explosion s e)
        (assoc-in [:eid->entity (:id e)] new-e))))




(def default-vehicle
  {:body {:collides? true
          :color defs/white
          :on-double-click-map
          {:die (fn [e s k]
                  {:updated-state (vehicle-death s e)})}
          :scale 0.4
          :stroke-weight 0}
   :components (concat
                [[:cart/motor :motor-bottom-right
                  {:anchor :bottom-right
                   :corner-r 5
                   :hidden? true
                   :on-update [(lib/->cap-activation)]
                   :rotational-power 0.02}]
                 [:cart/motor :motor-bottom-left
                  {:anchor :bottom-left
                   :corner-r 5
                   :hidden? true
                   :on-update [(lib/->cap-activation)]
                   :rotational-power 0.02}]])})

(def exploration-arousal
  [[:brain/neuron :exploration-arousal
    {;; :arousal 1
     :arousal-neuron? true
     :on-update [(fn [e _]
                   (update-in e
                              [:activation]
                              (fnil + 0)
                              (abs
                                (lib/normal-distr 1 1))))]}]
   [:brain/connection :_
    {:destination [:ref :motor-bottom-right]
     :f :excite
     :hidden? true
     :on-update-map
       {:gain (lib/every-n-seconds
                0.2
                (fn [e s k]
                  (assoc-in e
                    [:transduction-model :gain]
                    (+ 0.5 (lib/normal-distr 0.1 0.1)))))}
     :source [:ref :exploration-arousal]}]
   [:brain/connection :_
    {:destination [:ref :motor-bottom-left]
     :f :excite
     :hidden? true
     :on-update-map
       {:gain (lib/every-n-seconds
                0.2
                (fn [e s k]
                  (assoc-in e
                    [:transduction-model :gain]
                    (+ 0.5 (lib/normal-distr 0.1 0.1)))))}
     :source [:ref :exploration-arousal]}]])

(defn ray-sensor
  [id opts]
  [:cart/sensor id (merge {:modality :rays} opts)])

(defn gaussian
  [mu sigma]
  (fn [x]
    (let [z (/ (- x mu) sigma)]
      (* (/ 1 (* sigma (Math/sqrt (* 2 Math/PI))))
         (Math/exp (/ (* -1 z z) 2))))))

(def default-ray-sensors-top
  [[:cart/sensor :ray-top-left
    {:anchor :top-left :modality :rays}]
   [:cart/sensor :ray-top-right
    {:anchor :top-right :modality :rays}]])

(def thevalues (atom []))

(defn vehicle-4a-wires
  []
  (let [gain-gene {:mu (rand-int 20)
                   :sigma (inc (rand-int 50))
                   :x-val (rand-int 100)}]
    (update default-vehicle
            :components
            concat
            default-ray-sensors-top
            ;; exploration-arousal
            [[:brain/connection :_
              {:destination [:ref :motor-bottom-left]
               :f :excite
               :gain (fn [v]
                       (* (:x-val gain-gene)
                          ((gaussian (:mu gain-gene)
                                     (:sigma gain-gene))
                           v)))
               :hidden? true
               :source [:ref :ray-top-right]}]
             [:brain/connection :_
              {:destination [:ref :motor-bottom-right]
               :f :excite
               :gain (fn [v]
                       (* (:x-val gain-gene)
                          ((gaussian (:mu gain-gene)
                                     (:sigma gain-gene))
                           v)))
               :hidden? true
               :source [:ref :ray-top-left]}]])))


(defn vehilce-2b-wires
  []
  (-> default-vehicle
      (update :components
              concat
              default-ray-sensors-top
              exploration-arousal
              [[:brain/connection :_
                {:destination [:ref :motor-bottom-left]
                 :f :excite
                 :hidden? true
                 :source [:ref :ray-top-right]}]
               [:brain/connection :_
                {:destination [:ref :motor-bottom-right]
                 :f :excite
                 :hidden? true
                 :source [:ref :ray-top-left]}]])
      ;; (assoc-in
      ;;  [:body :color]
      ;;  (defs/color-map :navajo-white))
  ))


;; explore
(defn vehilce-3b-wires
  []
  (-> default-vehicle
      (update :components
              concat
              default-ray-sensors-top
              exploration-arousal
              [[:brain/connection :_
                {:destination [:ref :motor-bottom-left]
                 :f :inhibit
                 :hidden? true
                 :source [:ref :ray-top-right]}]
               [:brain/connection :_
                {:destination [:ref :motor-bottom-right]
                 :f :inhibit
                 :hidden? true
                 :source [:ref :ray-top-left]}]])
      ;; (assoc-in
      ;;  [:body :color]
      ;;  (defs/color-map :navajo-white))
      ))

;; love
(defn vehilce-3a-wires
  []
  (-> default-vehicle
      (update :components
              concat
              default-ray-sensors-top
              exploration-arousal
              [[:brain/connection :_
                {:destination [:ref :motor-bottom-left]
                 :f :inhibit
                 :hidden? true
                 :source [:ref :ray-top-left]}]
               [:brain/connection :_
                {:destination [:ref :motor-bottom-right]
                 :f :inhibit
                 :hidden? true
                 :source [:ref :ray-top-right]}]])))



;; (defn vehicle-1 ([] (vehicle-1 {})) ([body-opts] (let [cart (cart/->cart)] cart)))

#_(defn some-rand-environment-things
    [defs n]
    (let [stuff (repeatedly n
                            #(rand-nth [:temp-cold :temp-hot
                                        :organic-matter
                                        :oxygen]))
          ->make
          {:organic-matter
           (fn []
             (elib/->organic-matter
              {:odor {:decay-rate 2 :intensity 40}
               :pos (lib/rand-on-canvas-gauss 0.5)}))
           :oxygen (fn []
                     (elib/->oxygen
                      {:odor {:decay-rate 2 :intensity 40}
                       :pos (lib/rand-on-canvas-gauss
                             0.2)}))
           :temp-cold (fn []
                        (elib/->temperature-bubble-1
                         (rand-temperature-bubble defs :cold)))
           :temp-hot (fn []
                       (elib/->temperature-bubble-1
                        (rand-temperature-bubble defs
                                                 :hot)))}]
      (mapcat (fn [op] (op)) (map ->make stuff))))







;; (->cart plan)
